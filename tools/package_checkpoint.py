#!/usr/bin/env python3
"""Turn a training checkpoint into the distributable inference checkpoint.

    python tools/package_checkpoint.py TRAINING_CKPT -o OUT_DIR [--parts 2]

Three steps, in order:

1. **Strip the optimizer state.** A training checkpoint carries Adam moments
   that inference never reads -- on this model 6.25 GB of the 9.39 GB. Only
   ``model`` and a little metadata survive.
2. **Verify the weights are untouched.** Every tensor in the stripped file is
   hashed and compared with the same tensor in the source, so "unmodified fp32"
   is a measurement rather than a claim.
3. **Split and checksum.** GitHub caps a single file at 2 GB, so the result is
   cut into N parts with a SHA-256 of the joined whole beside them.

Do not add a dtype cast here. fp16 was measured and rejected: the text
projection holds values up to 9.58e18, past fp16's 65504 ceiling, so casting
turns those weights into ``inf`` and the model stops detecting anything.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam3_insect.weights import sha256sum, split_file  # noqa: E402

#: Keys worth carrying into the inference file. Everything else is training state.
KEEP = ("model", "epoch", "steps", "best_meter_values", "time_elapsed")


def _tensor_digest(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", help="Training checkpoint (.pt)")
    parser.add_argument("-o", "--out-dir", default="dist", help="Output directory")
    parser.add_argument(
        "--name", default="checkpoint_18_inference.pt", help="Output file name"
    )
    parser.add_argument("--parts", type=int, default=2, help="How many parts to split into")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the per-tensor hash comparison (it costs a second load)",
    )
    args = parser.parse_args(argv)

    import torch

    src = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.name

    print(f"Loading {src} ({src.stat().st_size / 2**30:.2f} GiB) ...")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    print("  keys:", ", ".join(sorted(ckpt)))

    slim = {k: ckpt[k] for k in KEEP if k in ckpt}
    if "model" not in slim:
        sys.exit("no 'model' key in the checkpoint — nothing to package")
    print(f"  keeping: {', '.join(slim)}  ({len(slim['model'])} tensors)")

    print(f"Writing {out_path} ...")
    torch.save(slim, out_path)
    print(f"  {out_path.stat().st_size / 2**30:.2f} GiB "
          f"({100 * out_path.stat().st_size / src.stat().st_size:.0f}% of the original)")

    if not args.skip_verify:
        print("Verifying every tensor against the source ...")
        check = torch.load(out_path, map_location="cpu", weights_only=False)["model"]
        source = ckpt["model"]
        assert set(check) == set(source), "tensor names changed"
        bad = [k for k in source if _tensor_digest(source[k]) != _tensor_digest(check[k])]
        if bad:
            sys.exit(f"{len(bad)} tensors differ, e.g. {bad[:3]}")
        print(f"  OK: all {len(source)} tensors hash-identical")
        del check, source

    del ckpt
    print(f"Splitting into {args.parts} parts ...")
    parts = split_file(out_path, n_parts=args.parts, out_dir=out_dir)
    for part in parts:
        print(f"  {part.name}  {part.stat().st_size} bytes "
              f"({part.stat().st_size / 2**30:.3f} GiB)")
        if part.stat().st_size >= 2 * 2**30:
            print("    WARNING: over GitHub's 2 GB per-file limit — use more parts")

    print(f"\nsha256: {sha256sum(out_path)}")
    print(f"Wrote {out_dir}/  — upload the parts and the .sha256 with "
          f"tools/publish_release.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
