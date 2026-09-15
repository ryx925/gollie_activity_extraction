"""Shared, conservative GoLLIE-7B 4-bit inference helpers."""

from __future__ import annotations

import ast
import gc
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_NAME = "HiTZ/GoLLIE-7B"
DEFAULT_MAX_NEW_TOKENS = 256

GUIDELINES = '''# The following lines describe the task definition
@dataclass
class MovementEvent(Template):
    """A concrete movement that actually occurred and involved a specific or
    identifiable person or group. Extract only movements supported by the text.
    Do not extract hypothetical, negated, merely planned, or generic historical
    movements. For example, do not extract "Merchants often travelled between
    cities." Never invent a missing participant, time, place, route, or mode.
    Preserve short verbatim evidence from the source text whenever possible."""

    person: str  # Person or identifiable group that moved; use exact source wording.
    activity: str  # Concrete movement action, preferably the original verb phrase.
    time: str | None  # Explicit time expression, otherwise None.
    location: str | None  # Explicit place associated with the movement, otherwise None.
    origin: str | None  # Explicit starting place, otherwise None.
    destination: str | None  # Explicit destination, otherwise None.
    transport_mode: str | None  # Explicit mode such as "by horse", otherwise None.
    evidence: str  # Short exact text span proving that the event occurred.


@dataclass
class ActivityEvent(Template):
    """A concrete activity that actually occurred and was performed by a
    specific or identifiable person or group. Do not extract hypothetical,
    negated, unfulfilled planned, habitual, or generic historical statements.
    Never hallucinate time or location. Preserve short verbatim evidence from
    the original text whenever possible."""

    person: str  # Person or identifiable group performing the activity.
    activity: str  # Actual performed activity, preferably the original verb phrase.
    time: str | None  # Explicit time expression, otherwise None.
    location: str | None  # Explicit activity location, otherwise None.
    origin: str | None  # Explicit origin if relevant, otherwise None.
    destination: str | None  # Explicit destination if relevant, otherwise None.
    transport_mode: str | None  # Explicit transport mode if relevant, otherwise None.
    evidence: str  # Short exact text span proving that the event occurred.
'''


def build_prompt(text: str) -> str:
    """Use the exact class/docstring/result-prefix protocol from GoLLIE's notebook."""
    return (
        f"{GUIDELINES}\n"
        "# This is the text to analyze\n"
        f"text = {text!r}\n\n"
        "# The annotation instances that take place in the text above are listed here\n"
        "result ="
    )


def gpu_memory_mib() -> dict[str, int | None]:
    result = {"used": None, "free": None, "total": None}
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        used, free, total = [int(x.strip()) for x in proc.stdout.splitlines()[0].split(",")]
        result.update(used=used, free=free, total=total)
    except Exception:
        pass
    return result


def clear_cuda_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_gollie():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to download or load GoLLIE")

    clear_cuda_memory()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def primary_input_device(model) -> Any:
    import torch

    device_map = getattr(model, "hf_device_map", {}) or {}
    for device in device_map.values():
        if device not in ("cpu", "disk"):
            return torch.device(f"cuda:{device}" if isinstance(device, int) else device)
    if getattr(model, "device", None) is not None:
        return model.device
    return torch.device("cuda:0")


@dataclass
class InferenceResult:
    output: str
    elapsed_seconds: float
    input_tokens: int
    output_tokens: int


def infer(model, tokenizer, text: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> InferenceResult:
    import torch

    prompt = build_prompt(text)
    inputs = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    # GoLLIE's official notebook removes a tokenizer-appended EOS before generation.
    if inputs["input_ids"].shape[1] and inputs["input_ids"][0, -1].item() == tokenizer.eos_token_id:
        inputs = {key: value[:, :-1] for key, value in inputs.items()}
    if inputs["input_ids"].shape[1] > 2000:
        raise ValueError(f"Prompt is {inputs['input_ids'].shape[1]} tokens; conservative limit is 2000")
    input_tokens = inputs["input_ids"].shape[1]
    input_device = primary_input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            min_new_tokens=0,
            num_beams=1,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    new_ids = generated[0, input_tokens:]
    output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return InferenceResult(output, elapsed, input_tokens, new_ids.numel())


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item) for item in node.elts]
    raise ValueError(f"unsupported value node: {type(node).__name__}")


def parse_events(output: str) -> list[dict[str, Any]]:
    """Safely parse simple GoLLIE constructor calls without executing model text."""
    candidate = output.strip()
    if not candidate.startswith("["):
        candidate = "[" + candidate
    # Generation can stop after the list; discard accidental trailing prose.
    match = re.search(r"\[.*\]", candidate, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
    try:
        tree = ast.parse(candidate, mode="eval")
    except SyntaxError:
        return []
    if not isinstance(tree.body, (ast.List, ast.Tuple)):
        return []
    parsed: list[dict[str, Any]] = []
    for node in tree.body.elts:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"MovementEvent", "ActivityEvent"}:
            continue
        event: dict[str, Any] = {"event_type": node.func.id}
        try:
            for keyword in node.keywords:
                if keyword.arg is not None:
                    event[keyword.arg] = _literal(keyword.value)
        except ValueError:
            continue
        parsed.append(event)
    return parsed


def describe_model(model) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "quantization": "bitsandbytes 4-bit NF4, double quantization, FP16 compute",
        "device_map": getattr(model, "hf_device_map", None),
        "gpu_memory_mib": gpu_memory_mib(),
    }

