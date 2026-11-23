#!/usr/bin/env python3
"""Write the lora_params.bin file that the evaluators in src/ read, from the adapter_model.safetensors of a particle.

usage: python scripts/convert_adapters.py <run_dir> [<run_dir> ...]
A run folder holds branch_0 to branch_{m-1}, one particle each. The tensors written to each lora_params.bin are
bitwise identical to the safetensors file next to it, which the script checks after writing.
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def convert(run_dir: Path) -> None:
    branches = sorted(run_dir.glob("branch_*"), key=lambda p: int(p.name.split("_")[1]))
    if not branches:
        sys.exit(f"error: no branch_* folders in {run_dir}")
    for branch in branches:
        state = load_file(branch / "adapter_model.safetensors")
        target = branch / "lora_params.bin"
        torch.save(state, target)
        back = torch.load(target, map_location="cpu", weights_only=True)
        if set(back) != set(state) or not all(torch.equal(back[k], state[k]) for k in state):
            sys.exit(f"error: {target} does not match {branch / 'adapter_model.safetensors'}")
    print(f"{run_dir}: lora_params.bin written for {len(branches)} particles")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        convert(Path(arg))
