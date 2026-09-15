"""Extract and conservatively chunk an EPUB without OCR."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

NOISE_TAGS = ("nav", "script", "style", "aside", "footer", "header", "form")
NOISE_CLASSES = re.compile(r"(footnote|endnote|page[-_ ]?number|running[-_ ]?head|toc|navigation)", re.I)


def clean_html(content: bytes) -> tuple[str | None, str]:
    soup = BeautifulSoup(content, "lxml")
    for node in soup.find_all(NOISE_TAGS):
        node.decompose()
    for node in soup.find_all(class_=NOISE_CLASSES):
        node.decompose()
    heading = soup.find(["h1", "h2", "h3"])
    title = " ".join(heading.get_text(" ", strip=True).split()) if heading else None
    paragraphs: list[str] = []
    for node in soup.find_all(["p", "blockquote", "li"]):
        text = " ".join(node.get_text(" ", strip=True).split())
        if len(text.split()) >= 3 and not NOISE_CLASSES.search(" ".join(node.get("class", []))):
            paragraphs.append(text)
    return title, "\n\n".join(paragraphs)


def extract_epub(epub_path: Path) -> dict[str, str]:
    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
    chapters: dict[str, str] = {}
    index = 1
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        name = (item.get_name() or "").lower()
        if any(marker in name for marker in ("nav", "toc", "cover")):
            continue
        title, text = clean_html(item.get_content())
        if len(text.split()) < 40:
            continue
        key = f"chapter_{index:02d}"
        display_title = title or f"Chapter {index}"
        chapters[key] = f"{display_title}\n\n{text}"
        index += 1
    if not chapters:
        raise RuntimeError("No substantial chapter text found in EPUB")
    return chapters


def paragraph_chunks(chapters: dict[str, str], target_words: int, overlap_words: int):
    for chapter_key, chapter_text in chapters.items():
        paragraphs = [p.strip() for p in chapter_text.split("\n\n") if p.strip()]
        chapter_title = paragraphs[0]
        body = paragraphs[1:]
        chunk_number = 1
        current: list[str] = []
        current_words = 0
        for paragraph in body:
            words = paragraph.split()
            if current and current_words + len(words) > target_words:
                yield {
                    "chunk_id": f"{chapter_key.replace('chapter_', 'ch')}_{chunk_number:03d}",
                    "chapter": chapter_title,
                    "text": "\n\n".join(current),
                }
                chunk_number += 1
                overlap: list[str] = []
                count = 0
                for old_paragraph in reversed(current):
                    overlap.insert(0, old_paragraph)
                    count += len(old_paragraph.split())
                    if count >= overlap_words:
                        break
                current = overlap
                current_words = sum(len(p.split()) for p in current)
            current.append(paragraph)
            current_words += len(words)
        if current:
            yield {
                "chunk_id": f"{chapter_key.replace('chapter_', 'ch')}_{chunk_number:03d}",
                "chapter": chapter_title,
                "text": "\n\n".join(current),
            }


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("epub", type=Path, help="Path to the source EPUB")
    parser.add_argument("--target-words", type=int, default=650)
    parser.add_argument("--overlap-words", type=int, default=125)
    args = parser.parse_args()
    if not 500 <= args.target_words <= 800:
        parser.error("--target-words must be between 500 and 800")
    if not 100 <= args.overlap_words <= 150:
        parser.error("--overlap-words must be between 100 and 150")

    chapters = extract_epub(args.epub.resolve())
    raw_dir = project_dir / "data" / "raw"
    chunk_dir = project_dir / "data" / "chunks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "book_by_chapter.json").write_text(
        json.dumps(chapters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (raw_dir / "book.txt").write_text("\n\n\n".join(chapters.values()), encoding="utf-8")
    chunks = list(paragraph_chunks(chapters, args.target_words, args.overlap_words))
    with (chunk_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Extracted {len(chapters)} chapters and {len(chunks)} chunks")
    print(f"Text:     {raw_dir / 'book.txt'}")
    print(f"Chapters: {raw_dir / 'book_by_chapter.json'}")
    print(f"Chunks:   {chunk_dir / 'chunks.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

