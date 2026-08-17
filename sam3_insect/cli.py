"""``sam3_insect_predict`` -- batch inference CLI.

    sam3_insect_predict -i IMAGE_OR_DIR -o OUTPUT_DIR [--weights PATH]

Writes ``coco_instances.json`` plus one overview image per input, mirroring the
layout flatbug's ``fb_predict`` produces so the two can be compared directly.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sam3_insect.coco import annotations_to_coco, save_coco
from sam3_insect.inference import DEFAULT_CFG, InsectPredictor, build_cfg
from sam3_insect.weights import resolve_checkpoint

IMAGE_PATTERN = r"[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG]|[tT][iI][fF]{1,2})$"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sam3_insect_predict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Detect and segment insects with the fine-tuned SAM 3 model.",
    )
    parser.add_argument("-i", "--input", required=True, help="Image file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        help="Checkpoint path. Omit to fetch it via --weights-source.",
    )
    parser.add_argument(
        "--weights-source",
        default="release",
        choices=["release", "hf", "lfs", "local"],
        help="Where to obtain the checkpoint when --weights is not given",
    )
    parser.add_argument(
        "--cache-dir", default="weights", help="Download cache for the checkpoint"
    )
    parser.add_argument("-p", "--pattern", default=IMAGE_PATTERN, help="Filename regex")
    parser.add_argument("-n", "--max-images", type=int, default=None)
    parser.add_argument("-R", "--recursive", action="store_true")
    parser.add_argument("-g", "--device", default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--config", default=None, help="YAML file of config overrides")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help=f"Override SCORE_THRESHOLD (default {DEFAULT_CFG['SCORE_THRESHOLD']}; "
        "lower it towards 0.005 only if you also raise it back when counting)",
    )
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--no-overviews", action="store_true")
    parser.add_argument("--per-image-json", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _collect_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.config:
        import yaml

        with open(args.config) as fh:
            overrides.update(yaml.safe_load(fh) or {})
    if args.score_threshold is not None:
        overrides["SCORE_THRESHOLD"] = args.score_threshold
    if args.mask_threshold is not None:
        overrides["MASK_THRESHOLD"] = args.mask_threshold
    return build_cfg(overrides)


def _collect_inputs(args: argparse.Namespace) -> List[str]:
    if os.path.isfile(args.input):
        return [args.input]
    files = sorted(
        f
        for f in glob.glob(os.path.join(args.input, "**"), recursive=args.recursive)
        if re.search(args.pattern, f)
    )
    return files[: args.max_images] if args.max_images else files


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    files = _collect_inputs(args)
    if not files:
        print(f"No images matched under {args.input}", file=sys.stderr)
        return 1

    cfg = _collect_cfg(args)

    if args.weights:
        checkpoint = resolve_checkpoint("local", path=args.weights)
    else:
        checkpoint = resolve_checkpoint(
            args.weights_source, cache_dir=args.cache_dir
        )

    predictor = InsectPredictor(
        checkpoint, device=args.device, cfg=cfg, verbose=args.verbose
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    overview_dir = out_dir / "overviews"
    if not args.no_overviews:
        overview_dir.mkdir(exist_ok=True)

    results = []
    print(f"Processing {len(files)} image(s) ...")
    for index, image_path in enumerate(files, start=1):
        try:
            result = predictor.predict(image_path)
        except Exception as exc:  # keep going through a bad file
            print(f"  [{index}/{len(files)}] ERROR {image_path}: {exc}", file=sys.stderr)
            continue

        print(f"  [{index}/{len(files)}] {result.file_name}: {len(result)} detections")
        results.append(result)

        if not args.no_overviews:
            stem = Path(result.file_name).stem
            predictor.render(result).save(
                overview_dir / f"overview_{stem}.jpg", quality=95
            )
        if args.per_image_json:
            save_coco(
                annotations_to_coco(result),
                out_dir / f"{Path(result.file_name).stem}.json",
            )

    coco_path = save_coco(annotations_to_coco(results), out_dir / "coco_instances.json")
    total = sum(len(r) for r in results)
    print(f"\nWrote {coco_path} — {len(results)} images, {total} annotations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
