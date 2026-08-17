# Model card — `checkpoint_18` (SAM 3 for Insects)

## What it is

Meta's **SAM 3** image model, fine-tuned for **detection and instance segmentation of terrestrial
arthropods** and served through flat-bug-style pyramid tiling. One instance mask, one bounding box
and one confidence per insect; output as COCO polygons.

| | |
|---|---|
| Base model | [SAM 3](https://github.com/facebookresearch/sam3) image model (ViT-L visual backbone, DETR-style decoder, 200 object queries) |
| Fine-tuning run | `flatbug_medium_ft`, **epoch 18** of 20 |
| Training data | The [flat-bug](https://github.com/darsa-group/flat-bug) aggregate — 23 insect datasets; 7,653 train / 978 validation images, 21,410 validation instances |
| Training resolution | 1008 px, random 600–1008 px crops at native resolution |
| Learning rate | 8e-5 |
| Batch size | 1 image per step |
| Prompt | Text: `"insect"` / `"insects"` |
| Classes | **One.** Arthropod vs. background — no taxonomy |
| Distributed weights | 3.14 GB, fp32, unmodified |

## Intended use

Counting and delineating arthropods in top-down imagery: bulk-sample trays, Petri dishes, pitfall
and light-trap catches, sticky cards, scanner plates, and field photographs of a handful of
specimens. It is a **research model** — useful for building datasets, pre-annotating, and
quantifying catches, given a human in the loop.

**Not intended for** species or genus identification, biomass estimation without calibration,
regulatory or pest-control decisions taken without review, or anything where a missed or invented
detection carries a real cost.

## How the weights are packaged

The training checkpoint is 9.39 GB, of which ~6.25 GB is Adam optimizer state inference never
reads. Only the model weights are distributed:

| | size |
|---|---|
| training checkpoint | 9.39 GB |
| ├─ `model` weights | 3.14 GB ← distributed |
| └─ `optimizer` state | 6.25 GB ← dropped |

Split into two ~1.57 GiB parts, since GitHub caps a single file at 2 GB. Compression was measured
and rejected: fp32 weights gzip to only ~2.91 GB, still over the limit.

**Fidelity, verified at three levels:**

- the reassembled file's SHA-256 matches the source file's
- all 1134 model tensors are hash-identical to the original 9.39 GB checkpoint
- the reassembled file loads with `torch.load` and reports `epoch: 18`

```
file    sha256: dd8a6ce0402a6c2d00b2849a3e08becc6f3aa4ececdc526580a54539c9c41829
weights sha256: 16644921001dec346e6953c8c0aee9175ea69d49d1eff048e91bc60e7aadef57
```

**Precision.** fp16 was evaluated and rejected: `backbone.language_backbone.encoder.text_projection`
holds values up to 9.58e18, past fp16's 65504 ceiling, so casting turns those weights into `inf`.
bf16 has the range but costs ~0.14% mean relative weight error. Exact fp32 was kept, and the
speed-up is taken at runtime instead: activations autocast to bfloat16 on GPUs that support it and
to float16 on older cards such as the T4.

## Known failure modes

**A false-positive floor at very low confidence, by construction.** The decoder spends a fixed
budget of 200 object queries in full on every tile and has no per-query "nothing here" output — a
query declines only by scoring low. On a tile with *N* insects you therefore get ~200 detections
above 0.005 and roughly *N* above 0.5, **at the same recall**. This is an operating-point property
of DETR-style detectors, not a hallucination introduced by fine-tuning, and it is why the library
defaults to a 0.02 threshold and the app filters its display at 0.4. Always fix a threshold before
reporting a count.

**Masks run tight.** Predicted masks tend to sit slightly inside the specimen outline relative to
flat-bug's ground truth, most visibly on hairy or translucent specimens and on legs and antennae.
Lower `MASK_THRESHOLD` towards 0.3, or add a few pixels of `MASK_DILATION_PIXELS`, when mask area
matters.

**Very small objects still need help.** Specimens spanning only a few pixels at full resolution
survive the pyramid poorly. Raise `SCALE_BEFORE` to 1.5–2.0 to give the model more pixels to work
with; coordinates are mapped back automatically.

**Domain shift costs precision more than recall.** On imagery unlike the training aggregate —
different backgrounds, extreme densities, unusual illumination — recall holds up better than
precision, so expect to raise the threshold rather than lower it.

**Overlapping and touching specimens.** Dense clusters, and insects lying across one another, are
merged or split inconsistently; NMS at IoU 0.2 is deliberately aggressive and will suppress a
genuinely overlapping pair.

**Cost.** ~3.2 GB of VRAM and roughly 10–60 s per image on a T4, scaling with megapixels because
the pyramid tile count does.

## Reproducibility notes

Two library defaults differ from the settings the model was originally measured with, both
deliberately and both reversible:

| | Library default | Original evaluation |
|---|---|---|
| `SCORE_THRESHOLD` | 0.02 | 0.005 |
| `EXIF_TRANSPOSE` | `True` | not applied |

```python
predictor.predict("img.jpg", cfg={"SCORE_THRESHOLD": 0.005, "EXIF_TRANSPOSE": False})
```

The EXIF difference is not cosmetic: some validation images carry a rotate-90 tag, so predictions
made with and without the transpose are in different coordinate frames.

## Licence and citation

The checkpoint is a derivative work of the SAM 3 materials, governed by the **SAM License** in
[`LICENSE`](LICENSE). Redistribution must include that licence and stay within Meta's Acceptable
Use Policy. Cite SAM 3 and flat-bug alongside this work — see the
[README](README.md#credits-and-licence).

Fine-tuning by Adam Basha, Karlsruhe Institute of Technology, 2026.
