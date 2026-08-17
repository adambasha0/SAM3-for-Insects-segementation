"""Fetching, splitting and verifying the fine-tuned checkpoint.

The fine-tuned weights are 3.14 GB of fp32 tensors, and GitHub caps a single
file at 2 GB, so the checkpoint ships as two ~1.57 GiB parts that this module
reassembles and checksums.  The split is a plain byte-wise concatenation, so
recovering the weights never depends on this code -- ``cat`` will do.

Four sources are supported:

``release``
    Two part files attached to a GitHub Release.  The default: plain HTTPS,
    no authentication, no rate limit.
``hf``
    A HuggingFace Hub model repo, if you mirror the weights to one.  Holds the
    checkpoint as a single resumable file, which is easier over a flaky link.
    Pass ``hf_repo=`` to point at your own.
``lfs``
    Part files present in a clone that tracks them through Git LFS.  Note that
    GitHub Free meters LFS bandwidth at 10 GiB/month, about three pulls.
``local``
    A path you already have: a Drive mount, a manual upload, or the original
    training checkpoint.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Union

PathLike = Union[str, "os.PathLike[str]"]

#: Size of the packaged inference checkpoint, in bytes.
INFERENCE_CKPT_BYTES = 3_371_878_637

#: SHA-256 of the reassembled inference checkpoint.
INFERENCE_CKPT_SHA256 = (
    "dd8a6ce0402a6c2d00b2849a3e08becc6f3aa4ececdc526580a54539c9c41829"
)

#: Default GitHub repo and release tag holding the part assets.
DEFAULT_GITHUB_REPO = "adambasha0/SAM3-for-Insects-segementation"
DEFAULT_RELEASE_TAG = "checkpoint-18"
DEFAULT_PART_NAMES = (
    "checkpoint_18_inference.pt.part-00",
    "checkpoint_18_inference.pt.part-01",
)

#: Default HuggingFace Hub location (publish there to use ``source="hf"``).
DEFAULT_HF_REPO = "adambasha0/sam3-for-insects-segmentation"
DEFAULT_HF_FILENAME = "checkpoint_18_inference.pt"


# ==========================================================================
# Hashing and splitting
# ==========================================================================


def sha256sum(path: PathLike, chunk_size: int = 8 << 20) -> str:
    """SHA-256 of a file, read in ``chunk_size`` blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_file(
    path: PathLike,
    n_parts: int = 2,
    out_dir: Optional[PathLike] = None,
    chunk_size: int = 8 << 20,
) -> List[Path]:
    """Split ``path`` into ``n_parts`` files named ``<name>.part-NN``.

    Used to prepare release assets; the reverse of :func:`join_parts`.
    """
    path = Path(path)
    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    total = path.stat().st_size
    # Ceil-divide so n_parts files always suffice.
    part_size = -(-total // n_parts)

    outputs: List[Path] = []
    with open(path, "rb") as src:
        for index in range(n_parts):
            out_path = out_dir / f"{path.name}.part-{index:02d}"
            written = 0
            with open(out_path, "wb") as dst:
                while written < part_size:
                    chunk = src.read(min(chunk_size, part_size - written))
                    if not chunk:
                        break
                    dst.write(chunk)
                    written += len(chunk)
            outputs.append(out_path)

    (out_dir / f"{path.name}.sha256").write_text(sha256sum(path) + "\n")
    return outputs


def join_parts(
    parts: Sequence[PathLike],
    out_path: PathLike,
    expected_sha256: Optional[str] = None,
    chunk_size: int = 8 << 20,
    remove_parts: bool = False,
) -> Path:
    """Concatenate ``parts`` into ``out_path`` and verify the checksum.

    Raises ``RuntimeError`` on a checksum mismatch, so a truncated download
    fails loudly instead of producing a checkpoint that loads to garbage.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts = [Path(p) for p in parts]
    for part in parts:
        if not part.exists() or part.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty part: {part}")
        with open(part, "rb") as fh:
            head = fh.read(200)
        if b"git-lfs.github.com" in head:
            raise RuntimeError(
                f"{part} is still a Git LFS pointer, not the real data. "
                "Run: git lfs install && git lfs pull"
            )

    print(f"Reassembling {len(parts)} parts -> {out_path}")
    with open(out_path, "wb") as dst:
        for part in parts:
            with open(part, "rb") as src:
                shutil.copyfileobj(src, dst, chunk_size)

    if expected_sha256:
        print("Verifying sha256 ...")
        actual = sha256sum(out_path)
        if actual != expected_sha256:
            out_path.unlink(missing_ok=True)
            raise RuntimeError(
                "checksum mismatch — the download is corrupt.\n"
                f"  expected {expected_sha256}\n  actual   {actual}"
            )
        print(f"OK: verified {out_path} ({out_path.stat().st_size} bytes).")

    if remove_parts:
        for part in parts:
            part.unlink(missing_ok=True)

    return out_path


# ==========================================================================
# Downloading
# ==========================================================================


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def download(
    url: str,
    out_path: PathLike,
    expected_size: Optional[int] = None,
    chunk_size: int = 8 << 20,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """Download ``url`` to ``out_path``, skipping a complete existing file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and (
        expected_size is None or out_path.stat().st_size == expected_size
    ):
        print(f"Already downloaded: {out_path.name} ({_human(out_path.stat().st_size)})")
        return out_path

    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "sam3-insect"})
    with urllib.request.urlopen(request) as response:  # noqa: S310 - fixed https URL
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
        with open(tmp_path, "wb") as fh:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                fraction = done / total if total else 0.0
                message = (
                    f"{out_path.name}: {_human(done)}"
                    + (f" / {_human(total)}" if total else "")
                )
                if progress is not None:
                    progress(fraction, message)
                else:
                    sys.stdout.write("\r  " + message)
                    sys.stdout.flush()
    if progress is None:
        sys.stdout.write("\n")
    tmp_path.replace(out_path)
    return out_path


def release_asset_url(
    asset: str,
    repo: str = DEFAULT_GITHUB_REPO,
    tag: str = DEFAULT_RELEASE_TAG,
) -> str:
    """URL of a GitHub Release asset."""
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


# ==========================================================================
# Source resolution
# ==========================================================================


def resolve_checkpoint(
    source: str = "release",
    *,
    path: Optional[PathLike] = None,
    cache_dir: PathLike = "weights",
    out_name: str = "checkpoint_18_inference.pt",
    repo: str = DEFAULT_GITHUB_REPO,
    tag: str = DEFAULT_RELEASE_TAG,
    part_names: Iterable[str] = DEFAULT_PART_NAMES,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_filename: str = DEFAULT_HF_FILENAME,
    repo_dir: Optional[PathLike] = None,
    expected_sha256: Optional[str] = INFERENCE_CKPT_SHA256,
    keep_parts: bool = False,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """Return a local path to the fine-tuned checkpoint, fetching if needed.

    Args:
        source: One of ``"release"``, ``"hf"``, ``"lfs"`` or ``"local"``.
        path: For ``source="local"``, the checkpoint path to use as-is.
        cache_dir: Where downloads and the reassembled file are kept.
        out_name: File name of the reassembled checkpoint.
        repo, tag, part_names: GitHub Release coordinates.
        hf_repo, hf_filename: HuggingFace Hub coordinates.
        repo_dir: Clone root for ``source="lfs"``; defaults to the repo that
            contains this file.
        expected_sha256: Checksum to enforce, or ``None`` to skip verification
            (necessary if you repackage the weights yourself).
        keep_parts: Keep the part files after joining.  They cost another
            3.14 GB of disk, which Colab can afford but rarely needs.
    """
    source = source.lower()
    cache_dir = Path(cache_dir)

    if source == "local":
        if path is None:
            raise ValueError('source="local" requires path=...')
        local = Path(os.fspath(path)).expanduser()
        if not local.is_file():
            raise FileNotFoundError(f"checkpoint not found: {local}")
        return local

    out_path = cache_dir / out_name
    if out_path.is_file() and out_path.stat().st_size == INFERENCE_CKPT_BYTES:
        print(f"Using cached checkpoint: {out_path}")
        return out_path

    if source == "hf":
        from huggingface_hub import hf_hub_download

        print(f"Fetching {hf_filename} from HuggingFace Hub repo {hf_repo} ...")
        return Path(
            hf_hub_download(
                repo_id=hf_repo,
                filename=hf_filename,
                local_dir=str(cache_dir),
            )
        )

    if source == "release":
        parts = [
            download(
                release_asset_url(name, repo=repo, tag=tag),
                cache_dir / name,
                progress=progress,
            )
            for name in part_names
        ]
        return join_parts(
            parts,
            out_path,
            expected_sha256=expected_sha256,
            remove_parts=not keep_parts,
        )

    if source == "lfs":
        root = Path(repo_dir) if repo_dir else Path(__file__).resolve().parent.parent
        part_dir = root / "models" / "checkpoint_18_inference"
        if not part_dir.is_dir():
            raise FileNotFoundError(f"expected LFS part directory at {part_dir}")

        print("Pulling Git LFS objects (counts against the repo's LFS bandwidth) ...")
        subprocess.run(["git", "lfs", "install"], cwd=root, check=True)
        subprocess.run(
            ["git", "lfs", "pull", "--include", "models/checkpoint_18_inference/*"],
            cwd=root,
            check=True,
        )
        return join_parts(
            [part_dir / name for name in part_names],
            out_path,
            expected_sha256=expected_sha256,
            remove_parts=False,  # they are tracked files; never delete them
        )

    raise ValueError(
        f'unknown source "{source}"; expected release, hf, lfs or local'
    )
