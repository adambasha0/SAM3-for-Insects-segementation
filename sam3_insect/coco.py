"""COCO assembly for predictions.

The emitted annotations carry both ``score`` and ``conf`` for the confidence,
because flatbug's evaluation tooling reads ``conf`` while most COCO consumers
expect ``score``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Sequence, Union

if TYPE_CHECKING:  # avoids a circular import at runtime
    from sam3_insect.inference import PredictionResult

CATEGORY_NAME = "insect"
CATEGORY_ID = 1


def _categories(category_id: int = CATEGORY_ID, name: str = CATEGORY_NAME):
    return [{"id": category_id, "name": name, "supercategory": "arthropod"}]


def empty_coco(
    category_id: int = CATEGORY_ID,
    category_name: str = CATEGORY_NAME,
    info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A COCO skeleton with no images or annotations."""
    return {
        "info": info or {"description": "sam3-insect predictions"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": _categories(category_id, category_name),
    }


def annotations_to_coco(
    results: Union["PredictionResult", Sequence["PredictionResult"]],
    category_id: int = CATEGORY_ID,
    category_name: str = CATEGORY_NAME,
    info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one COCO dict from one or more ``PredictionResult`` objects."""
    if not isinstance(results, (list, tuple)):
        results = [results]

    coco = empty_coco(category_id, category_name, info)
    ann_id = 1

    for image_id, result in enumerate(results, start=1):
        coco["images"].append(
            {
                "id": image_id,
                "file_name": result.file_name,
                "width": result.width,
                "height": result.height,
            }
        )
        for ann in result.annotations:
            score = float(ann["score"])
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(v) for v in ann["bbox"]],
                    "segmentation": [
                        [float(v) for v in poly] for poly in ann["segmentation"]
                    ],
                    "area": float(ann["area"]),
                    "iscrowd": 0,
                    "score": score,
                    "conf": score,  # flatbug compatibility
                }
            )
            ann_id += 1

    return coco


def merge_coco(parts: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge several COCO dicts, renumbering image and annotation ids."""
    merged: Optional[Dict[str, Any]] = None
    image_id = 1
    ann_id = 1

    for part in parts:
        if merged is None:
            merged = {
                "info": part.get("info", {}),
                "licenses": part.get("licenses", []),
                "images": [],
                "annotations": [],
                "categories": part.get("categories", _categories()),
            }
        id_map = {}
        for image in part.get("images", []):
            new_image = dict(image)
            id_map[image["id"]] = image_id
            new_image["id"] = image_id
            merged["images"].append(new_image)
            image_id += 1
        for ann in part.get("annotations", []):
            new_ann = dict(ann)
            new_ann["id"] = ann_id
            new_ann["image_id"] = id_map.get(ann["image_id"], ann["image_id"])
            merged["annotations"].append(new_ann)
            ann_id += 1

    return merged if merged is not None else empty_coco()


def save_coco(coco: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> Path:
    """Write a COCO dict to ``path`` and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco, indent=indent))
    return path
