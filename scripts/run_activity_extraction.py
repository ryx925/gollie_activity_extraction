"""Run a bounded GoLLIE activity extraction test over selected EPUB chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gollie_runtime import describe_model, infer, load_gollie, parse_events


def load_selected(path: Path, ids: set[str], limit: int) -> list[dict]:
    selected: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if ids and item["chunk_id"] not in ids:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break
    return selected


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=project_dir / "data" / "chunks" / "chunks.jsonl")
    parser.add_argument("--chunk-id", action="append", default=[], help="Repeat to select exact chunk IDs")
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 11))
    parser.add_argument("--max-new-tokens", type=int, default=256, choices=(128, 256, 384, 512))
    args = parser.parse_args()
    chunks = load_selected(args.chunks, set(args.chunk_id), args.limit)
    if not chunks:
        raise RuntimeError("No matching chunks found")

    model, tokenizer = load_gollie()
    print(json.dumps(describe_model(model), indent=2, ensure_ascii=False, default=str))
    output_path = project_dir / "data" / "outputs" / "activity_events_test.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, chunk in enumerate(chunks, start=1):
            print(f"[{index}/{len(chunks)}] {chunk['chunk_id']} ({len(chunk['text'].split())} words)")
            result = infer(model, tokenizer, chunk["text"], args.max_new_tokens)
            record = {
                "chunk_id": chunk["chunk_id"],
                "chapter": chunk["chapter"],
                "raw_text": chunk["text"],
                "raw_model_output": result.output,
                "parsed_events": parse_events(result.output),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "inference_seconds": round(result.elapsed_seconds, 3),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"  {result.elapsed_seconds:.2f}s; {len(record['parsed_events'])} parsed events")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

