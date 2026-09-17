"""Validate generated text and chunks against the source EPUB."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from extract_epub import (
    MAX_WORDS,
    MIN_WORDS,
    build_manifest,
    extract_selected_chapters,
    sentence_chunks,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preview(text: str, width: int = 100) -> tuple[str, str]:
    compact = " ".join(text.split())
    return compact[:width], compact[-width:]


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("epub", type=Path)
    args = parser.parse_args()
    epub_path = args.epub.resolve()

    chapter_path = project_dir / "data" / "raw" / "book_by_chapter.json"
    text_path = project_dir / "data" / "raw" / "book.txt"
    chunks_path = project_dir / "data" / "chunks" / "chunks.jsonl"
    manifest_path = project_dir / "data" / "chunks" / "chunk_manifest.json"

    chapters = extract_selected_chapters(epub_path)
    expected_chunks = sentence_chunks(chapters, epub_path.name)
    expected_chapters = {
        f"chapter_{chapter.number:02d}": {
            "chapter_number": chapter.number,
            "chapter": chapter.display_title,
            "source_href": chapter.source_href,
            "word_count": chapter.word_count,
            "text": chapter.text,
        }
        for chapter in chapters
    }
    expected_text = "\n\n\n".join(
        f"{chapter.display_title}\n\n{chapter.text}" for chapter in chapters
    )
    expected_manifest = build_manifest(epub_path, chapters, expected_chunks)

    actual_chapters = json.loads(chapter_path.read_text(encoding="utf-8"))
    actual_text = text_path.read_text(encoding="utf-8")
    actual_chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert actual_chapters == expected_chapters, "Chapter JSON does not match source EPUB"
    assert actual_text == expected_text, "book.txt does not match selected source chapters"
    assert actual_chunks == expected_chunks, "chunks.jsonl is not a deterministic source transform"
    assert actual_manifest == expected_manifest, "manifest does not match generated artifacts"
    assert [chapter.number for chapter in chapters] == list(range(1, 19))
    assert len(actual_chunks) == len({row["chunk_id"] for row in actual_chunks})

    for chapter in chapters:
        rows = [row for row in actual_chunks if row["chapter_number"] == chapter.number]
        for index, row in enumerate(rows):
            assert row["chapter"] == chapter.display_title
            assert 1 <= row["paragraph_start"] <= row["paragraph_end"] <= len(chapter.paragraphs)
            assert row["word_count"] == len(row["text"].split())
            assert row["word_count"] <= MAX_WORDS
            assert index == len(rows) - 1 or row["word_count"] >= MIN_WORDS
            assert row["text_sha256"] == hashlib.sha256(row["text"].encode("utf-8")).hexdigest()

    word_counts = [row["word_count"] for row in actual_chunks]
    overlap_counts = [row["overlap_words"] for row in actual_chunks if row["overlap_words"]]
    source_words = sum(chapter.word_count for chapter in chapters)
    chunk_words = sum(word_counts)
    sample_ids = ("ch01_001", "ch09_010", "ch17_010", "ch18_003")
    rows_by_id = {row["chunk_id"]: row for row in actual_chunks}

    print("QUALITY_CHECK=PASS")
    print(f"SOURCE_SHA256={file_sha256(epub_path)}")
    print(f"CHAPTERS={len(chapters)}")
    print(f"SOURCE_WORDS={source_words}")
    print(f"CHUNKS={len(actual_chunks)}")
    print(f"CHUNK_WORDS_WITH_OVERLAP={chunk_words}")
    print(f"OVERLAP_OVERHEAD_PERCENT={(chunk_words / source_words - 1) * 100:.2f}")
    print(
        "CHUNK_WORDS_MIN_MEDIAN_MAX="
        f"{min(word_counts)},{statistics.median(word_counts):.1f},{max(word_counts)}"
    )
    print(
        "NONZERO_OVERLAP_MIN_MEDIAN_MAX="
        f"{min(overlap_counts)},{statistics.median(overlap_counts):.1f},{max(overlap_counts)}"
    )
    print("ARTIFACTS")
    for path in (chapter_path, text_path, chunks_path, manifest_path):
        print(f"  {path.name}: {path.stat().st_size} bytes")
    print("BOUNDARY_SAMPLES")
    for chunk_id in sample_ids:
        row = rows_by_id[chunk_id]
        start, end = preview(row["text"])
        print(
            f"  {chunk_id} | {row['chapter']} | paragraphs "
            f"{row['paragraph_start']}-{row['paragraph_end']} | "
            f"words={row['word_count']} overlap={row['overlap_words']}"
        )
        print(f"    START: {start}")
        print(f"    END:   {end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
