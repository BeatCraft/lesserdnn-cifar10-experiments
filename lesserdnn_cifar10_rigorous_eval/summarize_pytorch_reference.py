"""Summarize the full-precision PyTorch CIFAR-10 reference."""

import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parent / "pytorch_reference_summary_core.py"
spec = importlib.util.spec_from_file_location("pytorch_reference_summary_core", path)
if spec is None or spec.loader is None:
    raise ImportError(f"could not load shared summarizer: {path}")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

if __name__ == "__main__":
    core.main()
