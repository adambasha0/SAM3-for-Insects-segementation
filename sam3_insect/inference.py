"""Pyramid-tiling inference for the fine-tuned SAM 3 insect model.

The tiling scheme, the pyramid scale ladder, the edge-case margin filter and the
polygon extraction are ports of ``flatbug``'s ``Localizer`` methodology, so that
predictions are directly comparable with flatbug's COCO output.

Two defaults differ from the runs the model was originally measured with, both
deliberately and both reversible:

* ``SCORE_THRESHOLD`` defaults to 0.02 rather than 0.005.  0.005 is the far left
  end of the PR curve and carries a large false-positive floor; 0.02 removes
  roughly 70% of those false positives for about 0.7pp of recall, which is the
  sane default for interactive use.
* ``EXIF_TRANSPOSE`` defaults to ``True`` so that phone photos are predicted in
  the orientation a viewer sees.  The original evaluation ran without it.
"""

from __future__ import annotations

import gc
import math
import os
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps

ImageLike = Union[str, "os.PathLike[str]", Image.Image, np.ndarray]


# ==========================================================================
# Configuration
# ==========================================================================

#: Defaults for the pyramid-tiling predictor.  Every key can be overridden per
#: predictor instance or per call via :func:`build_cfg`.
DEFAULT_CFG: Dict[str, Any] = {
    # --- tiling -----------------------------------------------------------
    "TILE_SIZE": 1024,
    "MINIMUM_TILE_OVERLAP": 384,
    "EDGE_CASE_MARGIN": 16,
    "IMAGE_BOUNDARY_MARGIN": 0,
    "SCALE_INCREMENT": 2 / 3,
    "PADDING": 32,
    "SCALE_BEFORE": 1.0,
    # --- detection --------------------------------------------------------
    "SCORE_THRESHOLD": 0.02,
    "IOU_THRESHOLD": 0.2,
    "MASK_THRESHOLD": 0.5,
    "MIN_MAX_OBJ_SIZE": (32, 1e8),
    "USE_IOS_NMS": False,
    # --- prompts ----------------------------------------------------------
    "PROMPT_PLURAL": "insects",
    "PROMPT_SINGULAR": "insect",
    "CATEGORY_ID": 1,
    # --- mask / polygon post-processing -----------------------------------
    "USE_CHAIN_APPROX_NONE": True,
    "USE_DYNAMIC_TOLERANCE": True,
    "LARGEST_CONTOUR_ONLY": True,
    "MASK_DILATION_PIXELS": 0,
    "MASK_EROSION_PIXELS": 0,
    "POLYGON_EXPANSION_PIXELS": 0,
    "BBOX_PADDING_PIXELS": 0,
    "MIN_MASK_AREA_PIXELS": 3,
    "LINEAR_INTERP_POINTS": 0,
    # --- input handling ---------------------------------------------------
    "EXIF_TRANSPOSE": True,
}


def build_cfg(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return :data:`DEFAULT_CFG` updated with ``overrides``.

    Unknown keys raise, so a typo in a YAML config fails loudly instead of
    being silently ignored.
    """
    cfg = dict(DEFAULT_CFG)
    if not overrides:
        return cfg
    unknown = sorted(set(overrides) - set(DEFAULT_CFG))
    if unknown:
        raise KeyError(
            f"unknown config key(s): {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(DEFAULT_CFG))}"
        )
    cfg.update(overrides)
    return cfg


@dataclass
class PredictionResult:
    """Predictions for one image, in original-image pixel coordinates."""

    annotations: List[Dict[str, Any]] = field(default_factory=list)
    width: int = 0
    height: int = 0
    file_name: str = "image"
    #: The image the coordinates refer to (post EXIF transpose), for rendering.
    image: Optional[Image.Image] = None

    def __len__(self) -> int:
        return len(self.annotations)

    @property
    def scores(self) -> List[float]:
        return [float(a["score"]) for a in self.annotations]


# ==========================================================================
# flatbug tiling primitives
# ==========================================================================


def equal_allocate_overlaps(total: int, segments: int, size: int) -> List[int]:
    """Cumulative start positions for ``segments`` tiles with even overlap."""
    if segments < 2:
        return [0] * segments

    overlap = segments * size - total
    partial_overlap, remainder = divmod(overlap, segments - 1)
    distance = size - partial_overlap

    return list(
        accumulate(
            [distance - (1 if i < remainder else 0) for i in range(segments - 1)],
            initial=0,
        )
    )


def calculate_tile_offsets(
    image_size: Tuple[int, int],
    tile_size: int,
    minimum_overlap: int,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Sliding-window tile offsets as ``((grid_m, grid_n), (y, x))`` pairs."""
    w, h = image_size

    x_n_tiles = (
        math.ceil((w - minimum_overlap) / (tile_size - minimum_overlap))
        if w != tile_size
        else 1
    )
    y_n_tiles = (
        math.ceil((h - minimum_overlap) / (tile_size - minimum_overlap))
        if h != tile_size
        else 1
    )

    x_range = equal_allocate_overlaps(w, x_n_tiles, tile_size)
    y_range = equal_allocate_overlaps(h, y_n_tiles, tile_size)

    return [
        ((m, n), (j, i))
        for n, j in enumerate(y_range)
        for m, i in enumerate(x_range)
    ]


def calculate_pyramid_scales(
    image_w: int,
    image_h: int,
    tile_size: int,
    scale_increment: float = 2 / 3,
) -> List[float]:
    """Pyramid scale ladder, from the coarsest whole-image view up to 1.0."""
    max_dim = max(image_w, image_h)
    scales: List[float] = []

    s = tile_size / max_dim
    if s >= 1.0:
        return [1.0]

    while s <= 0.9:
        scales.append(s)
        s /= scale_increment

    scales.append(1.0)
    return sorted(scales)


def filter_by_edge_margin(
    boxes: np.ndarray,
    tile_size: int,
    edge_margin: int,
    tile_x: int,
    tile_y: int,
    layer_w: int,
    layer_h: int,
) -> np.ndarray:
    """Drop detections touching a *synthetic* tile edge (a cut, not a border)."""
    if len(boxes) == 0:
        return np.array([], dtype=bool)

    is_real_left = tile_x == 0
    is_real_top = tile_y == 0
    is_real_right = tile_x + tile_size >= layer_w
    is_real_bottom = tile_y + tile_size >= layer_h

    keep = np.ones(len(boxes), dtype=bool)

    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box

        touches_left = x0 < edge_margin
        touches_top = y0 < edge_margin
        touches_right = x1 > tile_size - edge_margin
        touches_bottom = y1 > tile_size - edge_margin

        if touches_left and not is_real_left:
            keep[i] = False
        elif touches_top and not is_real_top:
            keep[i] = False
        elif touches_right and not is_real_right:
            keep[i] = False
        elif touches_bottom and not is_real_bottom:
            keep[i] = False

    return keep


def filter_by_image_boundary(
    boxes: np.ndarray,
    image_w: int,
    image_h: int,
    margin: int,
) -> np.ndarray:
    """Drop detections touching the real image boundary (disabled by default)."""
    if len(boxes) == 0:
        return np.array([], dtype=bool)
    if margin <= 0:
        return np.ones(len(boxes), dtype=bool)

    keep = np.ones(len(boxes), dtype=bool)
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        if (
            x0 < margin
            or y0 < margin
            or x1 > image_w - margin
            or y1 > image_h - margin
        ):
            keep[i] = False
    return keep


def filter_by_object_size(
    boxes: np.ndarray,
    min_size: float,
    max_size: float,
) -> np.ndarray:
    """Keep boxes whose ``sqrt(area)`` falls inside ``[min_size, max_size]``."""
    if len(boxes) == 0:
        return np.array([], dtype=bool)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    sizes = np.sqrt(widths * heights)
    return (sizes >= min_size) & (sizes <= max_size)


# ==========================================================================
# Mask / polygon post-processing
# ==========================================================================


def find_contours_flatbug(
    mask: np.ndarray,
    largest_only: bool = True,
    use_chain_approx_none: bool = True,
) -> List[np.ndarray]:
    """External contours of a binary mask, optionally the largest one only."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() == 1:
        mask = mask * 255

    approx_method = (
        cv2.CHAIN_APPROX_NONE if use_chain_approx_none else cv2.CHAIN_APPROX_SIMPLE
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, approx_method)
    if len(contours) == 0:
        return []

    if largest_only and len(contours) > 1:
        areas = [cv2.contourArea(c) for c in contours]
        contours = [contours[int(np.argmax(areas))]]

    return list(contours)


def calculate_dynamic_tolerance(
    mask_height: int,
    mask_width: int,
    image_height: int,
    image_width: int,
) -> float:
    """Polygon simplification tolerance scaled to the mask-to-image ratio."""
    scale_h = (image_height - 1) / max(mask_height - 1, 1)
    scale_w = (image_width - 1) / max(mask_width - 1, 1)
    return max((scale_h + scale_w) / 2 / 2, 1.0)


def _morph(mask: np.ndarray, kernel_size: int, op) -> np.ndarray:
    if kernel_size <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return op(mask, kernel, iterations=1)


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    return _morph(mask, kernel_size, cv2.dilate)


def erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    return _morph(mask, kernel_size, cv2.erode)


def expand_polygon(polygon: Sequence[float], expansion_px: float) -> List[float]:
    """Push polygon vertices outward from the centroid by ``expansion_px``."""
    polygon = list(polygon)
    if expansion_px <= 0 or len(polygon) < 6:
        return polygon

    pts = np.array(polygon).reshape(-1, 2)
    directions = pts - pts.mean(axis=0)
    distances = np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-6)
    return (pts + directions / distances * expansion_px).flatten().tolist()


def linear_interpolate_polygon(
    polygon: Sequence[float], n_interp: int = 10
) -> List[float]:
    """Insert ``n_interp`` points along every polygon edge."""
    polygon = list(polygon)
    if n_interp <= 0 or len(polygon) < 6:
        return polygon

    pts = np.array(polygon).reshape(-1, 2)
    n_pts = len(pts)

    interpolated = []
    for i in range(n_pts):
        p1, p2 = pts[i], pts[(i + 1) % n_pts]
        interpolated.append(p1)
        for j in range(1, n_interp + 1):
            interpolated.append(p1 + (j / (n_interp + 1)) * (p2 - p1))

    return np.array(interpolated).flatten().tolist()


def check_min_mask_area(mask: np.ndarray, min_area: int = 3) -> bool:
    """Whether a mask covers at least ``min_area`` pixels."""
    binary = (mask > 127) if mask.max() > 1 else mask
    return int(np.asarray(binary, dtype=np.uint8).sum()) >= min_area


def compute_ios_matrix(boxes: np.ndarray) -> np.ndarray:
    """Pairwise intersection-over-smaller matrix."""
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 0))

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    ios = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ix1, iy1 = max(x1[i], x1[j]), max(y1[i], y1[j])
            ix2, iy2 = min(x2[i], x2[j]), min(y2[i], y2[j])
            inter = (ix2 - ix1) * (iy2 - iy1) if ix2 > ix1 and iy2 > iy1 else 0.0
            smaller = min(areas[i], areas[j])
            value = inter / smaller if smaller > 0 else 0.0
            ios[i, j] = ios[j, i] = value
    return ios


def nms_ios(
    boxes: np.ndarray, scores: np.ndarray, ios_threshold: float = 0.2
) -> np.ndarray:
    """Greedy NMS using intersection-over-smaller instead of IoU."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(scores)[::-1]
    ios = compute_ios_matrix(boxes)

    keep: List[int] = []
    suppressed: set = set()
    for idx in order:
        if idx in suppressed:
            continue
        keep.append(int(idx))
        for other in order:
            if other != idx and other not in suppressed:
                if ios[idx, other] > ios_threshold:
                    suppressed.add(other)
    return np.array(keep, dtype=np.int64)


def pad_bbox(
    bbox: Sequence[float], padding: int, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Grow an ``[x0, y0, x1, y1]`` box, clamped to the image."""
    x0, y0, x1, y1 = bbox
    if padding <= 0:
        return x0, y0, x1, y1
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(img_w, x1 + padding),
        min(img_h, y1 + padding),
    )


def mask_to_polygons(
    mask_uint8: np.ndarray,
    x_off: int,
    y_off: int,
    scale: float,
    tile_size: int = 1024,
    use_dynamic_tolerance: bool = True,
    largest_only: bool = True,
    use_chain_approx_none: bool = True,
) -> List[List[float]]:
    """Contour a tile mask and map it into padded-image coordinates."""
    if use_dynamic_tolerance:
        mask_h, mask_w = mask_uint8.shape[:2]
        global_tile_size = int(tile_size / scale)
        tolerance = calculate_dynamic_tolerance(
            mask_h, mask_w, global_tile_size, global_tile_size
        )
    else:
        tolerance = 1.0

    polygons: List[List[float]] = []
    for cnt in find_contours_flatbug(
        mask_uint8,
        largest_only=largest_only,
        use_chain_approx_none=use_chain_approx_none,
    ):
        if len(cnt) < 3:
            continue

        points = cnt.reshape(-1, 2).astype(np.float64)
        points[:, 0] = (points[:, 0] + x_off) / scale
        points[:, 1] = (points[:, 1] + y_off) / scale
        points = np.maximum(points, 0)

        simplified = cv2.approxPolyDP(
            points.astype(np.float32).reshape(-1, 1, 2),
            epsilon=tolerance,
            closed=True,
        )
        if len(simplified) >= 3:
            polygons.append(simplified.reshape(-1).tolist())

    return polygons


# ==========================================================================
# Image loading
# ==========================================================================


def load_image(source: ImageLike, exif_transpose: bool = True) -> Image.Image:
    """Load ``source`` as RGB, honouring the EXIF orientation tag by default.

    Some images carry a rotate-90 tag, so predictions made with and without the
    transpose live in different coordinate frames.  For interactive use the
    transposed orientation is what the user sees, so it is applied by default.
    """
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, np.ndarray):
        image = Image.fromarray(source)
    else:
        image = Image.open(os.fspath(source))

    if exif_transpose:
        image = ImageOps.exif_transpose(image)

    return image.convert("RGB")


def pad_image(image: Image.Image, padding: int) -> Image.Image:
    """Letterbox ``image`` with ``padding`` black pixels on every side."""
    if padding <= 0:
        return image
    w, h = image.size
    padded = Image.new("RGB", (w + 2 * padding, h + 2 * padding), (0, 0, 0))
    padded.paste(image, (padding, padding))
    return padded


# ==========================================================================
# Predictor
# ==========================================================================


def _pick_autocast_dtype(device: torch.device) -> Optional[torch.dtype]:
    """bfloat16 where supported, float16 on older GPUs, none on CPU.

    Colab's free tier hands out Turing T4s, which have no bfloat16 units --
    autocasting to bf16 there is emulated and slow, so fall back to fp16.
    """
    if device.type != "cuda":
        return None
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


class InsectPredictor:
    """Fine-tuned SAM 3 insect detector with flatbug-style pyramid tiling.

    Args:
        checkpoint: Path to a checkpoint holding the fine-tuned weights, either
            the packaged inference file or a full training checkpoint.  A
            ``model`` / ``state_dict`` wrapper key is unwrapped automatically.
        device: Torch device string; defaults to CUDA when available.
        cfg: Overrides for :data:`DEFAULT_CFG`.
        bpe_path: Path to the CLIP BPE vocabulary shipped in ``assets/``.
        autocast_dtype: Force an autocast dtype instead of auto-selecting.
        strict: Whether the checkpoint must match the model exactly.
    """

    def __init__(
        self,
        checkpoint: Union[str, "os.PathLike[str]"],
        device: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        bpe_path: Optional[str] = None,
        autocast_dtype: Optional[torch.dtype] = None,
        strict: bool = True,
        verbose: bool = True,
    ) -> None:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.cfg = build_cfg(cfg)
        self.verbose = verbose

        # load_from_HF=False keeps us off the gated ``facebook/sam3`` repo: the
        # fine-tuned checkpoint carries every weight the model needs, so no
        # HuggingFace token or licence acceptance is required to run inference.
        if verbose:
            print("Building SAM 3 architecture (no HF download needed)...")
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device="cpu",
            load_from_HF=False,
        )

        if verbose:
            print(f"Loading fine-tuned weights from {checkpoint} ...")
        state = torch.load(os.fspath(checkpoint), map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            epoch = state.get("epoch")
            for key in ("model", "state_dict"):
                if key in state:
                    state = state[key]
                    break
        else:
            epoch = None

        missing, unexpected = model.load_state_dict(state, strict=strict)
        if verbose:
            if epoch is not None:
                print(f"  checkpoint epoch: {epoch}")
            if missing:
                print(f"  WARNING: {len(missing)} missing key(s), e.g. {missing[:3]}")
            if unexpected:
                print(
                    f"  WARNING: {len(unexpected)} unexpected key(s), "
                    f"e.g. {unexpected[:3]}"
                )

        del state
        gc.collect()

        model.to(self.device)
        model.eval()
        self.model = model
        self.processor = Sam3Processor(
            model,
            device=str(self.device),
            confidence_threshold=self.cfg["SCORE_THRESHOLD"],
        )
        self.autocast_dtype = (
            autocast_dtype
            if autocast_dtype is not None
            else _pick_autocast_dtype(self.device)
        )
        if verbose:
            dtype_name = (
                self.autocast_dtype
                and str(self.autocast_dtype).replace("torch.", "")
                or "fp32"
            )
            print(f"Ready on {self.device} (autocast: {dtype_name}).")

    # -- single tile -------------------------------------------------------

    def _infer_tile(
        self, tile: Image.Image, prompt: str, score_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the model on one tile, returning boxes, scores and mask logits."""
        autocast = (
            torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else torch.autocast(device_type=self.device.type, enabled=False)
        )
        with autocast, torch.inference_mode():
            state = self.processor.set_image(tile)
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(prompt, state)

        masks_logits = state.get("masks_logits", state.get("masks", []))
        boxes = state.get("boxes", [])
        scores = state.get("scores", [])

        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        b = boxes.float().detach().cpu().numpy()
        s = scores.float().detach().cpu().numpy().flatten()
        m = masks_logits.float().detach().cpu().numpy()
        if m.ndim == 4:
            m = m.squeeze(1)

        keep = s > score_threshold
        return b[keep], s[keep], m[keep]

    # -- whole image -------------------------------------------------------

    def predict(
        self,
        source: ImageLike,
        cfg: Optional[Dict[str, Any]] = None,
        file_name: Optional[str] = None,
        progress=None,
    ) -> PredictionResult:
        """Detect and segment insects in one image.

        Args:
            source: Path, PIL image or HWC uint8 array.
            cfg: Per-call config overrides on top of the instance config.
            file_name: Name recorded in the COCO ``images`` entry.
            progress: Optional ``callable(fraction, message)`` for UI progress.

        Returns:
            A :class:`PredictionResult` in original-image coordinates.
        """
        c = dict(self.cfg)
        if cfg:
            build_cfg(cfg)  # validates the keys, raises on typos
            c.update(cfg)

        tile_size = c["TILE_SIZE"]
        padding = c["PADDING"]
        min_size, max_size = c["MIN_MAX_OBJ_SIZE"]
        scale_before = c["SCALE_BEFORE"] or 1.0

        if file_name is None:
            file_name = (
                os.path.basename(os.fspath(source))
                if isinstance(source, (str, os.PathLike))
                else "image"
            )

        display_image = load_image(source, exif_transpose=c["EXIF_TRANSPOSE"])
        true_w, true_h = display_image.size

        # Optional pre-upscale so tiny objects span more model pixels; all
        # coordinates are divided back out before returning.
        work_image = display_image
        if scale_before != 1.0:
            work_image = display_image.resize(
                (round(true_w * scale_before), round(true_h * scale_before)),
                Image.Resampling.LANCZOS,
            )
        work_w, work_h = work_image.size

        padded_image = pad_image(work_image, padding)
        padded_w, padded_h = padded_image.size

        scales = calculate_pyramid_scales(
            padded_w, padded_h, tile_size, c["SCALE_INCREMENT"]
        )
        min_scale = min(scales)

        all_boxes: List[List[float]] = []
        all_scores: List[float] = []
        all_masks: List[Dict[str, Any]] = []

        # Count tiles up front so `progress` can report a real fraction.
        plan = []
        for scale in reversed(scales):
            if scale == 1.0:
                layer_size = (padded_w, padded_h)
            else:
                layer_size = (
                    round(padded_w * scale / 4) * 4,
                    round(padded_h * scale / 4) * 4,
                )
            offsets = calculate_tile_offsets(
                image_size=layer_size,
                tile_size=tile_size,
                minimum_overlap=int(c["MINIMUM_TILE_OVERLAP"] * scale),
            )
            plan.append((scale, layer_size, offsets))
        total_tiles = sum(len(offsets) for _, _, offsets in plan) or 1
        done_tiles = 0

        for scale, (layer_w, layer_h), offsets in plan:
            is_max_scale = scale == min_scale
            prompt = c["PROMPT_SINGULAR"] if is_max_scale else c["PROMPT_PLURAL"]

            layer_img = (
                padded_image
                if scale == 1.0
                else padded_image.resize((layer_w, layer_h), Image.Resampling.LANCZOS)
            )

            for _grid, (tile_y, tile_x) in offsets:
                done_tiles += 1
                if progress is not None:
                    progress(
                        done_tiles / total_tiles,
                        f"scale {scale:.2f} — tile {done_tiles}/{total_tiles}",
                    )

                tile = layer_img.crop(
                    (
                        tile_x,
                        tile_y,
                        min(tile_x + tile_size, layer_w),
                        min(tile_y + tile_size, layer_h),
                    )
                )
                if tile.size != (tile_size, tile_size):
                    canvas = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                    canvas.paste(tile, (0, 0))
                    tile = canvas

                t_boxes, t_scores, t_masks = self._infer_tile(
                    tile, prompt, c["SCORE_THRESHOLD"]
                )
                if len(t_boxes) == 0:
                    continue

                edge_keep = filter_by_edge_margin(
                    t_boxes,
                    tile_size,
                    0 if is_max_scale else c["EDGE_CASE_MARGIN"],
                    tile_x,
                    tile_y,
                    layer_w,
                    layer_h,
                )
                t_boxes, t_scores, t_masks = (
                    t_boxes[edge_keep],
                    t_scores[edge_keep],
                    t_masks[edge_keep],
                )
                if len(t_boxes) == 0:
                    continue

                size_keep = filter_by_object_size(
                    t_boxes, min_size, 1e9 if is_max_scale else max_size
                )
                t_boxes, t_scores, t_masks = (
                    t_boxes[size_keep],
                    t_scores[size_keep],
                    t_masks[size_keep],
                )

                for i in range(len(t_boxes)):
                    x0, y0, x1, y1 = t_boxes[i]
                    gx0 = max(0, (x0 + tile_x) / scale - padding)
                    gy0 = max(0, (y0 + tile_y) / scale - padding)
                    gx1 = min(work_w, (x1 + tile_x) / scale - padding)
                    gy1 = min(work_h, (y1 + tile_y) / scale - padding)

                    mask_bin = (t_masks[i] > c["MASK_THRESHOLD"]).astype(np.uint8)
                    if c["MIN_MASK_AREA_PIXELS"] > 0 and not check_min_mask_area(
                        mask_bin, c["MIN_MASK_AREA_PIXELS"]
                    ):
                        continue

                    mask_bin = erode_mask(mask_bin, c["MASK_EROSION_PIXELS"])
                    mask_bin = dilate_mask(mask_bin, c["MASK_DILATION_PIXELS"])

                    all_boxes.append([gx0, gy0, gx1, gy1])
                    all_scores.append(float(t_scores[i]))
                    all_masks.append(
                        {
                            "mask": mask_bin,
                            "x_off": tile_x,
                            "y_off": tile_y,
                            "scale": scale,
                            "padding": padding,
                            "tile_size": tile_size,
                        }
                    )

            if scale != 1.0:
                del layer_img
            gc.collect()

        final_boxes, final_scores, final_masks = self._suppress(
            all_boxes, all_scores, all_masks, c, work_w, work_h
        )

        annotations = self._to_annotations(
            final_boxes, final_scores, final_masks, c, work_w, work_h
        )

        if scale_before != 1.0:
            inv = 1.0 / scale_before
            for ann in annotations:
                ann["bbox"] = [v * inv for v in ann["bbox"]]
                ann["area"] *= inv * inv
                ann["segmentation"] = [
                    [v * inv for v in poly] for poly in ann["segmentation"]
                ]

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return PredictionResult(
            annotations=annotations,
            width=true_w,
            height=true_h,
            file_name=file_name,
            image=display_image,
        )

    # -- helpers -----------------------------------------------------------

    def _suppress(
        self,
        all_boxes: List[List[float]],
        all_scores: List[float],
        all_masks: List[Dict[str, Any]],
        c: Dict[str, Any],
        work_w: int,
        work_h: int,
    ):
        if not all_boxes:
            return np.array([]), np.array([]), []

        boxes_arr = np.array(all_boxes, dtype=np.float32)
        scores_arr = np.array(all_scores, dtype=np.float32)

        if c["USE_IOS_NMS"]:
            keep = nms_ios(boxes_arr, scores_arr, c["IOU_THRESHOLD"])
        else:
            keep = (
                torchvision.ops.nms(
                    torch.from_numpy(boxes_arr).to(self.device),
                    torch.from_numpy(scores_arr).to(self.device),
                    c["IOU_THRESHOLD"],
                )
                .cpu()
                .numpy()
            )

        final_boxes = boxes_arr[keep]
        final_scores = scores_arr[keep]
        final_masks = [all_masks[i] for i in keep]

        margin = c["IMAGE_BOUNDARY_MARGIN"]
        if margin > 0 and len(final_boxes) > 0:
            boundary_keep = filter_by_image_boundary(
                final_boxes, work_w, work_h, margin
            )
            final_boxes = final_boxes[boundary_keep]
            final_scores = final_scores[boundary_keep]
            final_masks = [
                m for m, k in zip(final_masks, boundary_keep) if k
            ]

        return final_boxes, final_scores, final_masks

    @staticmethod
    def _to_annotations(
        final_boxes,
        final_scores,
        final_masks,
        c: Dict[str, Any],
        work_w: int,
        work_h: int,
    ) -> List[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = []

        for idx in range(len(final_boxes)):
            m_info = final_masks[idx]
            polys = mask_to_polygons(
                m_info["mask"],
                m_info["x_off"],
                m_info["y_off"],
                m_info["scale"],
                tile_size=m_info["tile_size"],
                use_dynamic_tolerance=c["USE_DYNAMIC_TOLERANCE"],
                largest_only=c["LARGEST_CONTOUR_ONLY"],
                use_chain_approx_none=c["USE_CHAIN_APPROX_NONE"],
            )

            adjusted_polys: List[List[float]] = []
            for poly in polys:
                adjusted: List[float] = []
                for i in range(0, len(poly), 2):
                    x = max(0.0, min(float(work_w), float(poly[i] - m_info["padding"])))
                    y = max(
                        0.0, min(float(work_h), float(poly[i + 1] - m_info["padding"]))
                    )
                    adjusted.extend([x, y])
                if len(adjusted) < 6:
                    continue

                if c["LINEAR_INTERP_POINTS"] > 0:
                    adjusted = linear_interpolate_polygon(
                        adjusted, c["LINEAR_INTERP_POINTS"]
                    )
                if c["POLYGON_EXPANSION_PIXELS"] > 0:
                    adjusted = expand_polygon(adjusted, c["POLYGON_EXPANSION_PIXELS"])
                    adjusted = [
                        max(0.0, min(float(work_w if i % 2 == 0 else work_h), v))
                        for i, v in enumerate(adjusted)
                    ]
                adjusted_polys.append(adjusted)

            if not adjusted_polys:
                continue

            box = final_boxes[idx]
            x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            if c["BBOX_PADDING_PIXELS"] > 0:
                x0, y0, x1, y1 = pad_bbox(
                    [x0, y0, x1, y1], c["BBOX_PADDING_PIXELS"], work_w, work_h
                )

            annotations.append(
                {
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "segmentation": adjusted_polys,
                    "area": float((x1 - x0) * (y1 - y0)),
                    "score": float(final_scores[idx]),
                }
            )

        return annotations

    # -- convenience -------------------------------------------------------

    def render(self, result: PredictionResult, **kwargs) -> Image.Image:
        """Draw ``result`` on its source image (see :func:`~sam3_insect.viz`)."""
        from sam3_insect.viz import render_overview

        if result.image is None:
            raise ValueError("result has no attached image; re-run predict()")
        return render_overview(result.image, result.annotations, **kwargs)
