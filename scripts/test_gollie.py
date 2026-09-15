"""Strict GoLLIE-7B NF4 smoke test using the official guideline protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gollie_runtime import describe_model, infer, load_gollie, parse_events

TEST_TEXT = (
    "Giovanni left the inn early in the morning. He walked to the market, where he "
    "bought food. Later, he travelled by horse to Florence and visited a merchant."
)


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    output_path = project_dir / "data" / "outputs" / "smoke_test.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading HiTZ/GoLLIE-7B with strict trust_remote_code=True and 4-bit NF4...")
    model, tokenizer = load_gollie()
    details = describe_model(model)
    print(json.dumps(details, indent=2, ensure_ascii=False, default=str))
    if not any("cuda" in str(device) or isinstance(device, int) for device in (details["device_map"] or {}).values()):
        raise RuntimeError("GoLLIE did not place any model module on the RTX GPU")

    result = infer(model, tokenizer, TEST_TEXT, max_new_tokens=256)
    parsed = parse_events(result.output)
    report = (
        f"MODEL: HiTZ/GoLLIE-7B\n"
        f"QUANTIZATION: 4-bit NF4 + double quantization + FP16 compute\n"
        f"INPUT TOKENS: {result.input_tokens}\n"
        f"OUTPUT TOKENS: {result.output_tokens}\n"
        f"INFERENCE SECONDS: {result.elapsed_seconds:.2f}\n"
        f"\nRAW MODEL OUTPUT\n{result.output}\n"
        f"\nPARSED EVENTS\n{json.dumps(parsed, ensure_ascii=False, indent=2)}\n"
    )
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

