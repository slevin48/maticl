"""Export a TabICL PyTorch checkpoint to a portable format readable from MATLAB.

Produces two files:
  <out>.bin  little-endian float32 values of every tensor, concatenated
  <out>.idx  text index: a "config <key> <value>" line per config entry, then
             one "tensor <name> <offset> <ndims> <dim1> ... <dimN>" line per tensor
             (offset counted in float32 elements, dims in PyTorch order)

Tensors are written in PyTorch (row-major) order, so reading a 2-D [out, in]
weight into a MATLAB array of size [in, out] yields its transpose directly.

Usage:
  python tools/export_weights.py [--ckpt PATH] [--out weights/tabicl-classifier-v2]
"""

import argparse
from pathlib import Path

import numpy as np
import torch


DEFAULT_CKPT = "tabicl-classifier-v2-20260212.ckpt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="checkpoint path (downloaded from HF if omitted)")
    parser.add_argument("--out", default="weights/tabicl-classifier-v2")
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if ckpt_path is None:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(repo_id="jingang/TabICL", filename=DEFAULT_CKPT)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config, state = ckpt["config"], ckpt["state_dict"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    offset = 0
    with open(out.with_suffix(".bin"), "wb") as fbin, open(out.with_suffix(".idx"), "w") as fidx:
        for key, value in config.items():
            fidx.write(f"config {key} {value}\n")
        for name, tensor in state.items():
            arr = tensor.detach().cpu().to(torch.float32).numpy()
            arr = np.ascontiguousarray(arr).astype("<f4")
            fbin.write(arr.tobytes())
            dims = " ".join(str(d) for d in arr.shape) if arr.ndim else "1"
            fidx.write(f"tensor {name} {offset} {max(arr.ndim, 1)} {dims}\n")
            offset += arr.size

    print(f"Wrote {len(state)} tensors ({offset} floats) to {out.with_suffix('.bin')}")


if __name__ == "__main__":
    main()
