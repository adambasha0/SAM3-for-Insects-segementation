"""Importing the model must not require optional or training-only packages.

`sam3.model.sam3_image` reaches `sam3.train.data.sam3_image_dataset` through the
collator's `Datapoint`, and that module used to import `decord` at module scope
-- which made a video library a hard requirement of plain image inference and
broke a fresh Colab at `InsectPredictor(...)`. These tests pin the dependency
surface so the regression cannot come back silently.

They import only; the model is never built, so there is no GPU or 3.14 GB
checkpoint involved.
"""
import importlib
import importlib.util

import pytest

#: Packages that must NOT be needed to import the inference path. Each is either
#: import-guarded in a try/except or imported lazily inside the function that
#: uses it.
OPTIONAL = [
    "decord",              # video frame reading, training datasets only
    "torchcodec",          # video decoding
    "flash_attn_interface",  # FlashAttention-3 fast path
    "cc_torch",            # CUDA connected components
    "torch_generic_nms",   # CUDA NMS
    "skimage",             # CPU connected-components fallback
]

#: Modules the inference path imports unconditionally, so they belong in the
#: project's install_requires.
REQUIRED = ["einops", "scipy", "pycocotools", "cv2", "torch", "torchvision", "PIL"]


@pytest.mark.parametrize("module", REQUIRED)
def test_required_dependency_is_installed(module):
    assert importlib.util.find_spec(module) is not None, (
        f"{module} is imported unconditionally by the inference path but is not "
        "installed -- add it to pyproject.toml's dependencies"
    )


def test_model_builder_imports_without_optional_packages(monkeypatch):
    """Import the builder with every optional package hidden."""
    import builtins

    real_import = builtins.__import__
    blocked = tuple(OPTIONAL)

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ImportError(f"{name} is hidden by this test")
        return real_import(name, *args, **kwargs)

    for name in list(OPTIONAL):
        monkeypatch.delitem(__import__("sys").modules, name, raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    # Drop anything already imported so the import actually re-executes.
    import sys

    for mod in [m for m in sys.modules if m.startswith("sam3")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    importlib.import_module("sam3.model_builder")
    importlib.import_module("sam3.model.sam3_image_processor")


def test_public_api_is_importable():
    import sam3_insect

    for name in (
        "InsectPredictor",
        "PredictionResult",
        "DEFAULT_CFG",
        "resolve_checkpoint",
        "join_parts",
        "annotations_to_coco",
        "render_overview",
        "INFERENCE_CKPT_SHA256",
    ):
        assert hasattr(sam3_insect, name), f"sam3_insect.{name} went missing"


def test_cli_and_app_modules_import():
    importlib.import_module("sam3_insect.cli")
    importlib.import_module("sam3_insect.app")
