"""Check the isolated GoLLIE inference environment and NVIDIA GPU."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def nvidia_free_memory() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.used,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        free, used, total, driver = [x.strip() for x in result.stdout.splitlines()[0].split(",")]
        return f"{free} MiB free / {used} MiB used / {total} MiB total; driver {driver}"
    except Exception as exc:  # diagnostic script: report rather than hide failures
        return f"unavailable ({type(exc).__name__}: {exc})"


def main() -> int:
    failures: list[str] = []
    print(f"Python version:       {platform.python_version()} ({sys.executable})")
    if sys.version_info[:2] != (3, 10):
        failures.append("Python 3.10 is required")

    try:
        import torch

        print(f"PyTorch version:      {torch.__version__}")
        print(f"CUDA available:       {torch.cuda.is_available()}")
        print(f"CUDA runtime version: {torch.version.cuda}")
        if not torch.cuda.is_available():
            failures.append("PyTorch cannot access CUDA")
        else:
            props = torch.cuda.get_device_properties(0)
            print(f"GPU name:             {torch.cuda.get_device_name(0)}")
            print(f"GPU total memory:     {props.total_memory / 1024**3:.2f} GiB")
            print(f"GPU capability:       {props.major}.{props.minor}")
            # A real CUDA allocation catches several DLL/runtime failures that a
            # simple is_available() check can miss.
            value = (torch.ones(1, device="cuda") * 2).item()
            print(f"CUDA tensor test:     {'PASS' if value == 2 else 'FAIL'}")
            if value != 2:
                failures.append("CUDA tensor computation failed")
    except Exception as exc:
        print(f"PyTorch error:        {type(exc).__name__}: {exc}")
        failures.append("PyTorch import or CUDA test failed")

    print(f"NVIDIA memory:        {nvidia_free_memory()}")
    print(f"Transformers version: {package_version('transformers')}")
    print(f"Accelerate version:   {package_version('accelerate')}")
    print(f"bitsandbytes version: {package_version('bitsandbytes')}")
    try:
        import bitsandbytes as bnb

        print(f"bitsandbytes import:  PASS ({bnb.__version__})")
    except Exception as exc:
        print(f"bitsandbytes import:  FAIL ({type(exc).__name__}: {exc})")
        failures.append("bitsandbytes import failed")

    if package_version("transformers") == "NOT INSTALLED":
        failures.append("transformers is not installed")

    print()
    if failures:
        print("ENVIRONMENT FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("ENVIRONMENT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

