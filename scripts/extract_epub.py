"""Extract the selected EPUB chapters and create auditable, sentence-aware chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ebooklib
from bs4 import BeautifulSoup, Tag
from ebooklib import epub

NOISE_TAGS = ("nav", "script", "style", "aside", "footer", "header", "form")
NOISE_CLASSES = re.compile(
    r"(footnote|endnote|page[-_ ]?number|running[-_ ]?head|toc|navigation)", re.I
)
CHAPTER_TITLE = re.compile(r"^(?P<number>[1-9]|1[0-8])\s+(?P<title>\S.*)$")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘([])")

TARGET_WORDS = 650
MIN_WORDS = 500
MAX_WORDS = 750
OVERLAP_TARGET_WORDS = 120
OVERLAP_MAX_WORDS = 150


@dataclass(frozen=True)
class TextUnit:
    text: str
    paragraph: int
    sentence: int
    section: str | None

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Chapter:
    number: int
    title: str
    source_href: str
    paragraphs: list[tuple[str | None, str]]

    @property
    def display_title(self) -> str:
        return f"{self.number} {self.title}"

    @property
    def text(self) -> str:
        return "\n\n".join(text for _, text in self.paragraphs)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def is_nested_content(node: Tag) -> bool:
    return node.find_parent(["p", "blockquote", "li"]) is not None


def clean_document(content: bytes) -> tuple[str | None, list[tuple[str | None, str]]]:
    """Return the first heading and ordered paragraphs with their active section."""
    soup = BeautifulSoup(content, "lxml-xml")
    for node in soup.find_all(NOISE_TAGS):
        node.decompose()
    for node in soup.find_all(class_=NOISE_CLASSES):
        node.decompose()

    first_heading: str | None = None
    active_section: str | None = None
    paragraphs: list[tuple[str | None, str]] = []
    for node in soup.find_all(["h1", "h2", "h3", "p", "blockquote", "li"]):
        if not isinstance(node, Tag):
            continue
        text = normalize_text(node.get_text(" ", strip=True))
        if not text:
            continue
        if node.name in {"h1", "h2", "h3"}:
            if first_heading is None:
                first_heading = text
            elif text != first_heading:
                active_section = text
            continue
        if is_nested_content(node):
            continue
        if len(text.split()) >= 3:
            paragraphs.append((active_section, text))
    return first_heading, paragraphs


def extract_selected_chapters(epub_path: Path) -> list[Chapter]:
    """Read formal chapters 1-18 in EPUB spine order."""
    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
    chapters: list[Chapter] = []
    seen_numbers: set[int] = set()
    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        heading, paragraphs = clean_document(item.get_content())
        match = CHAPTER_TITLE.fullmatch(heading or "")
        if not match:
            continue
        number = int(match.group("number"))
        if number in seen_numbers:
            raise RuntimeError(f"Duplicate formal chapter number: {number}")
        if not paragraphs:
            raise RuntimeError(f"Formal chapter {number} contains no body paragraphs")
        seen_numbers.add(number)
        chapters.append(
            Chapter(
                number=number,
                title=match.group("title"),
                source_href=item.get_name(),
                paragraphs=paragraphs,
            )
        )
    expected = list(range(1, 19))
    actual = [chapter.number for chapter in chapters]
    if actual != expected:
        raise RuntimeError(f"Expected formal chapters {expected}; found {actual}")
    return chapters


def split_long_words(words: list[str], limit: int = OVERLAP_MAX_WORDS) -> Iterable[str]:
    for start in range(0, len(words), limit):
        yield " ".join(words[start : start + limit])


def chapter_units(chapter: Chapter) -> list[TextUnit]:
    units: list[TextUnit] = []
    for paragraph_number, (section, paragraph) in enumerate(chapter.paragraphs, start=1):
        sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(paragraph) if part.strip()]
        sentence_number = 1
        for sentence in sentences:
            words = sentence.split()
            pieces = [sentence] if len(words) <= OVERLAP_MAX_WORDS else list(split_long_words(words))
            for piece in pieces:
                units.append(TextUnit(piece, paragraph_number, sentence_number, section))
                sentence_number += 1
    return units


def select_overlap(units: list[TextUnit]) -> list[TextUnit]:
    selected: list[TextUnit] = []
    total = 0
    for unit in reversed(units):
        if total + unit.word_count > OVERLAP_MAX_WORDS:
            if not selected:
                words = unit.text.split()[-OVERLAP_MAX_WORDS:]
                selected.insert(0, TextUnit(" ".join(words), unit.paragraph, unit.sentence, unit.section))
            break
        selected.insert(0, unit)
        total += unit.word_count
        if total >= OVERLAP_TARGET_WORDS:
            break
    return selected


def render_units(units: list[TextUnit]) -> str:
    paragraphs: list[str] = []
    current_paragraph: int | None = None
    current: list[str] = []
    for unit in units:
        if current and unit.paragraph != current_paragraph:
            paragraphs.append(" ".join(current))
            current = []
        current_paragraph = unit.paragraph
        current.append(unit.text)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def make_chunk(chapter: Chapter, number: int, units: list[TextUnit], overlap_words: int) -> dict:
    text = render_units(units)
    sections = list(dict.fromkeys(unit.section for unit in units if unit.section))
    return {
        "chunk_id": f"ch{chapter.number:02d}_{number:03d}",
        "source_file": None,
        "source_href": chapter.source_href,
        "chapter_number": chapter.number,
        "chapter": chapter.display_title,
        "sections": sections,
        "paragraph_start": min(unit.paragraph for unit in units),
        "paragraph_end": max(unit.paragraph for unit in units),
        "word_count": len(text.split()),
        "overlap_words": overlap_words,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
    }


def sentence_chunks(chapters: list[Chapter], source_file: str) -> list[dict]:
    chunks: list[dict] = []
    for chapter in chapters:
        units = chapter_units(chapter)
        current: list[TextUnit] = []
        current_words = 0
        leading_overlap_words = 0
        packed: list[tuple[list[TextUnit], int]] = []
        for unit in units:
            next_words = current_words + unit.word_count
            should_flush = bool(current) and (
                (current_words >= MIN_WORDS and next_words > TARGET_WORDS)
                or next_words > MAX_WORDS
            )
            if should_flush:
                packed.append((current, leading_overlap_words))
                current = select_overlap(current)
                current_words = sum(item.word_count for item in current)
                leading_overlap_words = current_words
            current.append(unit)
            current_words += unit.word_count
        if current:
            packed.append((current, leading_overlap_words))

        if len(packed) > 1 and sum(unit.word_count for unit in packed[-1][0]) < MIN_WORDS:
            final_units, final_overlap = packed[-1]
            prefix_words = 0
            first_unique = 0
            while first_unique < len(final_units) and prefix_words < final_overlap:
                prefix_words += final_units[first_unique].word_count
                first_unique += 1
            previous_units, previous_overlap = packed[-2]
            merged_units = previous_units + final_units[first_unique:]
            if sum(unit.word_count for unit in merged_units) <= MAX_WORDS:
                packed[-2] = (merged_units, previous_overlap)
                packed.pop()

        chapter_chunks = [
            make_chunk(chapter, number, chunk_units, overlap_words)
            for number, (chunk_units, overlap_words) in enumerate(packed, start=1)
        ]
        for chunk in chapter_chunks:
            chunk["source_file"] = source_file
        chunks.extend(chapter_chunks)
    return chunks


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(epub_path: Path, chapters: list[Chapter], chunks: list[dict]) -> dict:
    chapter_rows = []
    for chapter in chapters:
        chapter_chunks = [row for row in chunks if row["chapter_number"] == chapter.number]
        chapter_rows.append(
            {
                "chapter_number": chapter.number,
                "chapter": chapter.display_title,
                "source_href": chapter.source_href,
                "source_words": chapter.word_count,
                "paragraphs": len(chapter.paragraphs),
                "chunks": len(chapter_chunks),
            }
        )
    return {
        "source_file": epub_path.name,
        "source_sha256": source_sha256(epub_path),
        "selection": "Formal chapters 1-18 only; front matter, timeline, bibliography, index, and author biography excluded.",
        "chunking": {
            "strategy": "sentence-aware, paragraph-preserving, no cross-chapter chunks",
            "target_words": TARGET_WORDS,
            "minimum_words": MIN_WORDS,
            "maximum_words": MAX_WORDS,
            "overlap_target_words": OVERLAP_TARGET_WORDS,
            "overlap_max_words": OVERLAP_MAX_WORDS,
        },
        "chapter_count": len(chapters),
        "source_words": sum(chapter.word_count for chapter in chapters),
        "chunk_count": len(chunks),
        "chunk_words_with_overlap": sum(row["word_count"] for row in chunks),
        "chapters": chapter_rows,
    }


def validate(chapters: list[Chapter], chunks: list[dict]) -> None:
    if [chapter.number for chapter in chapters] != list(range(1, 19)):
        raise RuntimeError("Formal chapter selection is incomplete or out of order")
    ids = [row["chunk_id"] for row in chunks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Chunk IDs are not unique")
    for chapter in chapters:
        rows = [row for row in chunks if row["chapter_number"] == chapter.number]
        expected_ids = [f"ch{chapter.number:02d}_{index:03d}" for index in range(1, len(rows) + 1)]
        if [row["chunk_id"] for row in rows] != expected_ids:
            raise RuntimeError(f"Chunk IDs are not contiguous for chapter {chapter.number}")
        for index, row in enumerate(rows):
            if not row["text"].strip():
                raise RuntimeError(f"Empty text in {row['chunk_id']}")
            if row["word_count"] > MAX_WORDS:
                raise RuntimeError(f"Oversized chunk {row['chunk_id']}: {row['word_count']} words")
            if index < len(rows) - 1 and row["word_count"] < MIN_WORDS:
                raise RuntimeError(f"Undersized non-final chunk {row['chunk_id']}")
            digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            if digest != row["text_sha256"]:
                raise RuntimeError(f"Text hash mismatch in {row['chunk_id']}")


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("epub", type=Path, help="Path to the source EPUB")
    args = parser.parse_args()
    epub_path = args.epub.resolve()

    chapters = extract_selected_chapters(epub_path)
    chunks = sentence_chunks(chapters, epub_path.name)
    validate(chapters, chunks)
    manifest = build_manifest(epub_path, chapters, chunks)

    raw_dir = project_dir / "data" / "raw"
    chunk_dir = project_dir / "data" / "chunks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chapter_payload = {
        f"chapter_{chapter.number:02d}": {
            "chapter_number": chapter.number,
            "chapter": chapter.display_title,
            "source_href": chapter.source_href,
            "word_count": chapter.word_count,
            "text": chapter.text,
        }
        for chapter in chapters
    }
    (raw_dir / "book_by_chapter.json").write_text(
        json.dumps(chapter_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (raw_dir / "book.txt").write_text(
        "\n\n\n".join(f"{chapter.display_title}\n\n{chapter.text}" for chapter in chapters),
        encoding="utf-8",
    )
    with (chunk_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    (chunk_dir / "chunk_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Extracted {len(chapters)} formal chapters and {len(chunks)} chunks")
    print(f"Source words: {manifest['source_words']}")
    print(f"Chunk words with overlap: {manifest['chunk_words_with_overlap']}")
    print(f"Text:     {raw_dir / 'book.txt'}")
    print(f"Chapters: {raw_dir / 'book_by_chapter.json'}")
    print(f"Chunks:   {chunk_dir / 'chunks.jsonl'}")
    print(f"Manifest: {chunk_dir / 'chunk_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
