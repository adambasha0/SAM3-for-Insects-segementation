"""Overview rendering: filled masks, outlines, boxes and score labels."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

MASK_FILL_COLOR_RGBA = (135, 206, 250, 60)
MASK_BORDER_COLOR_RGBA = (0, 81, 255, 220)
BBOX_COLOR = "#0051FF"
LABEL_TEXT_COLOR = "black"


def _load_font(size: int) -> Optional[ImageFont.ImageFont]:
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 9.2 has no size argument
        return ImageFont.load_default()


def _auto_scale(image: Image.Image) -> Tuple[int, int, int]:
    """Line width, mask border width and font size proportional to image size."""
    reference = max(image.size)
    line_width = max(2, round(reference / 640))
    border_width = max(2, round(reference / 480))
    font_size = max(12, round(reference / 55))
    return line_width, border_width, font_size


def render_overview(
    image: Image.Image,
    annotations: Sequence[Dict[str, Any]],
    draw_masks: bool = True,
    draw_boxes: bool = True,
    draw_scores: bool = True,
    score_threshold: float = 0.0,
    line_width: Optional[int] = None,
    font_size: Optional[int] = None,
) -> Image.Image:
    """Return a copy of ``image`` with predictions drawn on top.

    ``annotations`` are COCO-style dicts with ``bbox`` in ``[x, y, w, h]``,
    ``segmentation`` as a list of flat polygons and a ``score``.
    """
    base = image.convert("RGBA")
    auto_line, auto_border, auto_font = _auto_scale(base)
    line_width = line_width or auto_line
    border_width = auto_border
    font = _load_font(font_size or auto_font)

    kept = [a for a in annotations if float(a.get("score", 1.0)) >= score_threshold]

    if draw_masks:
        mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_layer)
        for ann in kept:
            for poly in ann.get("segmentation", []):
                if len(poly) < 6:
                    continue
                points = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                mask_draw.polygon(points, fill=MASK_FILL_COLOR_RGBA)
                mask_draw.line(
                    points + [points[0]],
                    fill=MASK_BORDER_COLOR_RGBA,
                    width=border_width,
                )
        base = Image.alpha_composite(base, mask_layer)

    out = base.convert("RGB")
    draw = ImageDraw.Draw(out)

    if draw_boxes:
        for ann in kept:
            x0, y0, w, h = ann["bbox"]
            x1, y1 = x0 + w, y0 + h
            draw.rectangle([x0, y0, x1, y1], outline=BBOX_COLOR, width=line_width)

            if not draw_scores:
                continue

            label = f"insect {float(ann.get('score', 0.0)):.2f}"
            text_box = draw.textbbox((0, 0), label, font=font)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]

            label_y = y0 - text_h - 6
            if label_y < 0:
                label_y = y0 + 4
            draw.rectangle(
                [x0, label_y, x0 + text_w + 8, label_y + text_h + 6],
                fill=BBOX_COLOR,
            )
            draw.text((x0 + 4, label_y + 2), label, fill=LABEL_TEXT_COLOR, font=font)

    return out


def crop_detections(
    image: Image.Image,
    annotations: Iterable[Dict[str, Any]],
    padding: int = 8,
    min_size: int = 8,
) -> List[Image.Image]:
    """Crop each detection out of ``image``, with a little context padding."""
    w, h = image.size
    crops: List[Image.Image] = []
    for ann in annotations:
        x0, y0, bw, bh = ann["bbox"]
        if bw < min_size or bh < min_size:
            continue
        box = (
            max(0, int(x0 - padding)),
            max(0, int(y0 - padding)),
            min(w, int(x0 + bw + padding)),
            min(h, int(y0 + bh + padding)),
        )
        if box[2] > box[0] and box[3] > box[1]:
            crops.append(image.crop(box))
    return crops
