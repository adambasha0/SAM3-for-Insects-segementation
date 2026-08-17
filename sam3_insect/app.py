"""Gradio front-end: upload an image, press Predict, download COCO.

Used by the Colab notebook (``docs/sam3_insect_colab.ipynb``) and runnable
locally:

    python -m sam3_insect.app --weights weights/checkpoint_18_inference.pt

Inference runs once at a low score threshold; the **Confidence** slider then
re-filters the cached detections, so moving it is instant instead of costing
another pass over the image pyramid.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from sam3_insect.coco import annotations_to_coco, save_coco
from sam3_insect.inference import InsectPredictor, PredictionResult
from sam3_insect.viz import crop_detections, render_overview

#: Inference runs at this threshold so the slider can go this low without a
#: re-run.  Below ~0.02 the model produces a large false-positive floor.
RUN_SCORE_THRESHOLD = 0.02
DEFAULT_CONFIDENCE = 0.40

#: Caps so that a dense trap image at low confidence cannot stall the browser
#: or spend minutes writing thousands of crop files.  The COCO export is never
#: capped -- it always contains every kept detection.
MAX_GALLERY_CROPS = 60
MAX_ZIP_CROPS = 300

DESCRIPTION = """
# SAM 3 for insects — interactive demo

Fine-tuned **SAM 3** (`flatbug_medium_ft`, epoch 18) for detection and instance
segmentation of terrestrial arthropods, run through flatbug-style pyramid
tiling so that small insects in large trap images are still found.

**How to use:** upload an image (or pick an example) → press **Predict** →
adjust **Confidence** to taste → download the COCO annotations.
"""

FOOTER = f"""
Inference runs once at a score threshold of {RUN_SCORE_THRESHOLD}; the
confidence slider filters those cached detections. Masks are exported as COCO
polygons, boxes as `[x, y, width, height]`; each annotation carries the
confidence in both `score` and `conf` (the latter for flatbug's evaluation
tools).
"""


# ==========================================================================
# Result formatting
# ==========================================================================


def _filter(result: PredictionResult, confidence: float) -> List[Dict[str, Any]]:
    return [a for a in result.annotations if float(a["score"]) >= confidence]


def _summary(result: PredictionResult, kept: Sequence[Dict[str, Any]], confidence: float) -> str:
    if not result.annotations:
        return "No detections at all — try a lower confidence or a closer crop."
    text = (
        f"**{len(kept)} detection(s)** at confidence ≥ {confidence:.2f} "
        f"({len(result.annotations)} before filtering) · "
        f"image {result.width}×{result.height}"
    )
    if len(kept) > MAX_GALLERY_CROPS:
        text += (
            f"\n\n*Crops tab shows the top {MAX_GALLERY_CROPS}; "
            "the COCO export has all of them.*"
        )
    return text


def _table(kept: Sequence[Dict[str, Any]]) -> List[List[Any]]:
    rows = []
    for index, ann in enumerate(
        sorted(kept, key=lambda a: -float(a["score"])), start=1
    ):
        x, y, w, h = (round(float(v), 1) for v in ann["bbox"])
        rows.append([index, round(float(ann["score"]), 3), x, y, w, h, round(float(ann["area"]))])
    return rows


def _write_outputs(
    result: PredictionResult,
    kept: Sequence[Dict[str, Any]],
    overview: Image.Image,
    out_dir: Path,
) -> Tuple[str, str]:
    """Write the COCO JSON and a zip bundle; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.file_name).stem or "image"

    filtered = PredictionResult(
        annotations=list(kept),
        width=result.width,
        height=result.height,
        file_name=result.file_name,
    )
    coco_path = save_coco(
        annotations_to_coco(filtered), out_dir / f"{stem}_coco.json"
    )

    overview_path = out_dir / f"{stem}_overview.jpg"
    overview.convert("RGB").save(overview_path, quality=95)

    zip_path = out_dir / f"{stem}_sam3_insect.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(coco_path, coco_path.name)
        zf.write(overview_path, overview_path.name)
        if result.image is not None:
            top = sorted(kept, key=lambda a: -float(a["score"]))[:MAX_ZIP_CROPS]
            for index, crop in enumerate(crop_detections(result.image, top), start=1):
                crop_path = out_dir / f"crop_{index:03d}.png"
                crop.save(crop_path)
                zf.write(crop_path, f"crops/{crop_path.name}")
                crop_path.unlink()

    return str(coco_path), str(zip_path)


# ==========================================================================
# Gradio app
# ==========================================================================


def build_demo(
    predictor: InsectPredictor,
    examples: Optional[Sequence[str]] = None,
    work_dir: Optional[str] = None,
):
    """Build the Gradio ``Blocks`` app around a loaded ``predictor``."""
    import gradio as gr

    work_root = Path(work_dir or tempfile.mkdtemp(prefix="sam3_insect_"))

    def run(image_path, confidence, mask_threshold, show_masks, show_boxes,
            show_scores, progress=gr.Progress()):
        if not image_path:
            raise gr.Error("Upload an image first.")

        def report(fraction, message):
            progress(fraction, desc=message)

        result = predictor.predict(
            image_path,
            cfg={
                "SCORE_THRESHOLD": RUN_SCORE_THRESHOLD,
                "MASK_THRESHOLD": float(mask_threshold),
            },
            progress=report,
        )
        return refresh(result, confidence, show_masks, show_boxes, show_scores)

    def refresh(result, confidence, show_masks, show_boxes, show_scores):
        if result is None:
            raise gr.Error("Run Predict first.")

        confidence = float(confidence)
        kept = _filter(result, confidence)
        overview = render_overview(
            result.image,
            kept,
            draw_masks=show_masks,
            draw_boxes=show_boxes,
            draw_scores=show_scores,
        )
        gallery = (
            crop_detections(
                result.image,
                sorted(kept, key=lambda a: -float(a["score"]))[:MAX_GALLERY_CROPS],
            )
            if result.image
            else []
        )
        coco_path, zip_path = _write_outputs(
            result, kept, overview, work_root / "outputs"
        )
        return (
            overview,
            gallery,
            _table(kept),
            _summary(result, kept, confidence),
            coco_path,
            zip_path,
            result,
        )

    # No explicit theme: Gradio 6 moved the argument to launch(), and the
    # default theme renders fine in both light and dark Colab.
    with gr.Blocks(title="SAM 3 for insects") as demo:
        gr.Markdown(DESCRIPTION)
        state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=4):
                image_input = gr.Image(
                    label="Input image", type="filepath", height=380
                )
                predict_button = gr.Button("Predict", variant="primary", size="lg")
                confidence = gr.Slider(
                    RUN_SCORE_THRESHOLD,
                    0.99,
                    value=DEFAULT_CONFIDENCE,
                    step=0.01,
                    label="Confidence",
                    info="Filters cached detections — no re-run needed",
                )
                with gr.Accordion("Display", open=False):
                    show_masks = gr.Checkbox(True, label="Segmentation masks")
                    show_boxes = gr.Checkbox(True, label="Bounding boxes")
                    show_scores = gr.Checkbox(True, label="Score labels")
                with gr.Accordion("Advanced (re-runs inference)", open=False):
                    mask_threshold = gr.Slider(
                        0.05,
                        0.95,
                        value=predictor.cfg["MASK_THRESHOLD"],
                        step=0.05,
                        label="Mask threshold",
                        info="Lower grows masks, higher tightens them",
                    )

            with gr.Column(scale=6):
                summary = gr.Markdown("Upload an image and press **Predict**.")
                with gr.Tabs():
                    with gr.Tab("Prediction"):
                        overview_output = gr.Image(
                            label="Detections", type="pil", height=520
                        )
                    with gr.Tab("Crops"):
                        gallery_output = gr.Gallery(
                            label="Cropped detections", columns=5, height=520
                        )
                    with gr.Tab("Detections"):
                        table_output = gr.Dataframe(
                            headers=["#", "score", "x", "y", "w", "h", "area"],
                            label="Detections (highest score first)",
                            wrap=True,
                        )
                with gr.Row():
                    coco_file = gr.File(label="COCO annotations (.json)")
                    zip_file = gr.File(label="Bundle (.zip): COCO + overview + crops")

        if examples:
            gr.Examples(examples=[[e] for e in examples], inputs=[image_input])

        gr.Markdown(FOOTER)

        run_inputs = [
            image_input,
            confidence,
            mask_threshold,
            show_masks,
            show_boxes,
            show_scores,
        ]
        outputs = [
            overview_output,
            gallery_output,
            table_output,
            summary,
            coco_file,
            zip_file,
            state,
        ]
        predict_button.click(run, inputs=run_inputs, outputs=outputs)

        refresh_inputs = [state, confidence, show_masks, show_boxes, show_scores]
        for control in (confidence, show_masks, show_boxes, show_scores):
            control.change(refresh, inputs=refresh_inputs, outputs=outputs)

    return demo


def launch(
    weights: str,
    device: Optional[str] = None,
    examples: Optional[Sequence[str]] = None,
    share: bool = True,
    **launch_kwargs,
):
    """Load the model and launch the app (``share=True`` suits Colab)."""
    predictor = InsectPredictor(weights, device=device)
    demo = build_demo(predictor, examples=examples)
    demo.queue().launch(share=share, **launch_kwargs)
    return demo


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the insect demo app.")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--examples", nargs="*", default=None)
    parser.add_argument("--share", action="store_true", help="Public gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)

    launch(
        args.weights,
        device=args.device,
        examples=args.examples,
        share=args.share,
        server_port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
