from __future__ import annotations

import os
import argparse


def configure_cuda(cuda_device: str = "0") -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_device)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def require_rtx3060(enabled: bool = True) -> dict:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required before checking the RTX3060 device") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install an NVIDIA CUDA-enabled PyTorch build and run on the RTX3060 host."
        )
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    total_gb = torch.cuda.get_device_properties(index).total_memory / 1024**3
    info = {
        "cuda_device": index,
        "cuda_name": name,
        "cuda_capability": f"{capability[0]}.{capability[1]}",
        "cuda_memory_gb": round(total_gb, 2),
    }
    normalized_name = name.upper().replace("GEFORCE ", "").replace(" ", "")
    if enabled and "RTX3060" not in normalized_name:
        raise RuntimeError(f"Expected RTX3060/RTX 3060 but current CUDA device is: {name}")
    print(
        "Using CUDA device "
        f"{info['cuda_device']}: {info['cuda_name']} "
        f"capability={info['cuda_capability']} memory={info['cuda_memory_gb']}GB",
        flush=True,
    )
    return info


def prepare_rtx3060(cuda_device: str = "0", require_device: bool = True) -> dict:
    configure_cuda(cuda_device)
    return require_rtx3060(require_device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose CUDA/RTX3060 availability for recall45 fine-tuning")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--allow-non-rtx3060", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_rtx3060(args.cuda_device, require_device=not args.allow_non_rtx3060)


if __name__ == "__main__":
    main()
