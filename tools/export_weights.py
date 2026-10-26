"""Export a TabICL PyTorch checkpoint to a portable format readable from MATLAB.

Produces two files:
  <out>.bin  little-endian float32 values of every tensor, concatenated
  <out>.idx  text index: a "config <key> <value>" line per config entry, an
             "archive <dir>" line, then one
             "tensor <name> <offset> <ndims> <dim1> ... <dimN> <storage>" line per
             tensor (offset counted in float32 elements, dims in PyTorch order,
             storage = name of the raw file <dir>/data/<storage> holding the
             tensor inside the checkpoint zip)

The index alone (committed as assets/tabicl-classifier-v2.idx) is enough for
tabicl_load to read the tensors straight from the unzipped Hugging Face
checkpoint, so MATLAB users do not need Python.

Tensors are written in PyTorch (row-major) order, so reading a 2-D [out, in]
weight into a MATLAB array of size [in, out] yields its transpose directly.

Usage:
  python tools/export_weights.py [--ckpt PATH] [--out weights/tabicl-classifier-v2]
"""

import argparse
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import torch


DEFAULT_CKPT = "tabicl-classifier-v2-20260212.ckpt"


def storage_keys(ckpt_path):
    """Map tensor bytes (sha256) to the raw storage file holding them in the zip.

    Every TabICL tensor owns a full, contiguous little-endian float32 storage,
    so a tensor's bytes equal its storage file. Identical tensors (e.g. zero
    biases) may map to either of their files, which is harmless.
    """
    storage_of = {}
    with zipfile.ZipFile(ckpt_path) as z:
        byteorder = [n for n in z.namelist() if n.endswith("/byteorder")]
        archive = byteorder[0].split("/")[0]
        assert z.read(byteorder[0]).decode().strip() == "little"
        for info in z.infolist():
            parts = info.filename.split("/")
            if len(parts) == 3 and parts[1] == "data":
                assert info.compress_type == zipfile.ZIP_STORED
                storage_of[hashlib.sha256(z.read(info)).hexdigest()] = parts[2]
    return storage_of, archive


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

    storage_of, archive = storage_keys(ckpt_path)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    offset = 0
    with open(out.with_suffix(".bin"), "wb") as fbin, open(out.with_suffix(".idx"), "w") as fidx:
        for key, value in config.items():
            fidx.write(f"config {key} {value}\n")
        fidx.write(f"archive {archive}\n")
        for name, tensor in state.items():
            arr = tensor.detach().cpu().to(torch.float32).numpy()
            arr = np.ascontiguousarray(arr).astype("<f4")
            fbin.write(arr.tobytes())
            dims = " ".join(str(d) for d in arr.shape) if arr.ndim else "1"
            storage = storage_of[hashlib.sha256(arr.tobytes()).hexdigest()]
            fidx.write(f"tensor {name} {offset} {max(arr.ndim, 1)} {dims} {storage}\n")
            offset += arr.size

    assets = Path(__file__).resolve().parents[1] / "assets"
    assets.mkdir(exist_ok=True)
    (assets / out.with_suffix(".idx").name).write_text(out.with_suffix(".idx").read_text())
    print(f"Wrote {len(state)} tensors ({offset} floats) to {out.with_suffix('.bin')}")


if __name__ == "__main__":
    main()
