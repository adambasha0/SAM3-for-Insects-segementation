"""Fine-tuned SAM 3 for detection and instance segmentation of insects.

This package wraps the ``flatbug_medium_ft`` checkpoint in a flatbug-style
pyramid-tiling inference scheme, so a single image (or a directory of images)
can be turned into COCO annotations and an overview rendering in a few lines:

    from sam3_insect import InsectPredictor, resolve_checkpoint

    ckpt = resolve_checkpoint("local", path="checkpoint_18_inference.pt")
    predictor = InsectPredictor(ckpt)
    result = predictor.predict("my_photo.jpg")
    overview = predictor.render(result)
"""

from sam3_insect.coco import annotations_to_coco, merge_coco
from sam3_insect.inference import (
    DEFAULT_CFG,
    InsectPredictor,
    PredictionResult,
    build_cfg,
)
from sam3_insect.viz import render_overview
from sam3_insect.weights import (
    INFERENCE_CKPT_BYTES,
    INFERENCE_CKPT_SHA256,
    join_parts,
    resolve_checkpoint,
    sha256sum,
    split_file,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_CFG",
    "INFERENCE_CKPT_BYTES",
    "INFERENCE_CKPT_SHA256",
    "InsectPredictor",
    "PredictionResult",
    "annotations_to_coco",
    "build_cfg",
    "join_parts",
    "merge_coco",
    "render_overview",
    "resolve_checkpoint",
    "sha256sum",
    "split_file",
    "__version__",
]
