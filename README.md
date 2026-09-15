# GoLLIE Activity Extraction (Windows / RTX 3060 6GB)

This project is a bounded inference-only setup for `HiTZ/GoLLIE-7B`. It never
loads the model in full FP16/BF16 and defaults to bitsandbytes 4-bit NF4,
double quantization, FP16 compute, `device_map="auto"`, batch size 1, no
sampling, and at most 256 newly generated tokens.

## Environment

- OS: Windows 11 64-bit
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6GB)
- NVIDIA driver: 596.08 (unchanged)
- Python: 3.10.11, isolated in `.venv`
- PyTorch: 2.7.1+cu128; CUDA tensor computation passed
- PyTorch CUDA runtime: 12.8
- Transformers: 4.44.2
- Accelerate: 0.34.2
- bitsandbytes: 0.48.1; actual 4-bit NF4 CUDA forward passed
- NumPy: 2.2.6; packaging: 24.2
- `nvidia-smi` at environment check: 4701 MiB free / 1294 MiB used

Core inference packages were installed and verified. A later audit found that
the background dependency installation also finished: EbookLib,
beautifulsoup4, and lxml are currently installed, but the EPUB has still not
been read or tested. For the exact current package state, use `pip list`; some
installed auxiliary versions differ from `requirements.txt`. The standalone
Chinese handoff is `GoLLIE_handoff_for_ChatGPT.md`.

## Commands

Run these from PowerShell in this directory:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_env.py
.\.venv\Scripts\python.exe .\scripts\test_gollie.py
.\.venv\Scripts\python.exe .\scripts\extract_epub.py "C:\path\to\book.epub"
.\.venv\Scripts\python.exe .\scripts\run_activity_extraction.py --limit 5
```

The smoke test uses GoLLIE's official class/docstring annotation-guideline
prompt shape and begins generation immediately after `result =`. Generated
Python-like constructors are parsed with `ast`; model output is never executed.

## Windows compatibility status

GoLLIE's model card requests `trust_remote_code=True`. Its remote
`modeling_flash_llama.py` imports FlashAttention 2 and raises `ImportError` when
that dependency is missing. As checked during setup, the official
flash-attention v2.8.3.post1 release publishes Linux wheels but no Windows
wheels. This machine also has no standalone CUDA Toolkit (`nvcc`) or MSVC build
toolchain, so a native source build is not currently available. A real
Transformers remote-class preflight imported the official 46KB GoLLIE file and
failed with `ModuleNotFoundError: No module named 'flash_attn'`, then
`ImportError: Please install Flash Attention: pip install flash-attn --no-build-isolation`.
The official 7B weights were not downloaded. The smoke test is intentionally
strict and does not silently fall back to another model or to
`trust_remote_code=False`. Consequently, model load, smoke inference, EPUB
extraction, and book-chunk inference have **not** been run.

The current recommended next route is WSL2 with the official FlashAttention
and GoLLIE Transformers+bitsandbytes path, subject to confirming WSL2 GPU
passthrough and available RAM. A separately approved fallback would be a
community GoLLIE-7B GGUF with llama.cpp, which is not the same official
Transformers+NF4 test and may yield different outputs.

### WSL2 transition status (2026-09-15, WSL 2.9.12 installed)

The user approved switching to WSL2 and rebooted after elevated DISM enabled
`Microsoft-Windows-Subsystem-Linux` and `VirtualMachinePlatform` (both initially
returned `3010`). An elevated post-reboot check confirmed both features are
`Enabled`. However, the WSL runtime is still not usable: `wsl --status` exits
`50`, `wsl --list --verbose` exits `1`, and `wsl --version` asks for WSL
installation. The user subsequently installed Microsoft WSL 2.9.12 x64.
`wsl --version` now exits 0 and reports WSL 2.9.12.0; `WslService` is Running;
an elevated `wsl --status` exits 0. In the current restricted command session,
`wsl --status` and `wsl --list --verbose` instead fail with
`Wsl/EnumerateDistros/Service/E_ACCESSDENIED`. No per-user distro registration
was found in `HKCU\Software\Microsoft\Windows\CurrentVersion\Lxss`. Thus the
WSL application is installed, but ordinary-session distro access and Ubuntu/GPU
passthrough remain unverified. Per the user's request, no further setup was run.
Windows `nvidia-smi` still detects the RTX 3060, driver 596.08, 6144 MiB VRAM.

The Microsoft-signed 17,104,896-byte legacy kernel updater downloaded
successfully, but elevated MSI installation exited `1603`. Its verbose log identifies the
failed launch condition: `This update only applies to machines with the Windows
Subsytem for Linux`. This is not a PyTorch, bitsandbytes, or GoLLIE failure.
Microsoft's current offline instructions call for the newer WSL MSI from its
GitHub releases. A 2.7.14 x64 MSI download was started but stopped after only
about 5.8 MiB because throughput was roughly 40-50 KiB/s; the file
`../wsl.2.7.14.0.x64.msi` was incomplete. After the user installed 2.9.12,
these three obsolete workspace artifacts (the old kernel MSI, its install log,
and the partial 2.7.14 MSI) were deleted. A direct HEAD request to that release
also timed out. Do not install a Linux NVIDIA
display driver. No GoLLIE weights have been downloaded, and no Linux Python
environment, FlashAttention, smoke inference, EPUB extraction, or book-chunk
test has been run. See `GoLLIE_handoff_for_ChatGPT.md` for the current handoff.

## EPUB output

Place or reference the legally obtained EPUB explicitly. Extraction uses
EbookLib plus BeautifulSoup/lxml (no OCR), preserves paragraph boundaries and
chapter labels, filters common navigation/footnote noise, and writes:

- `data/raw/book.txt`
- `data/raw/book_by_chapter.json`
- `data/chunks/chunks.jsonl` (650 words, paragraph-aware, 125-word overlap)

The bounded book test writes `data/outputs/activity_events_test.jsonl` and
accepts at most 10 chunks per invocation.
