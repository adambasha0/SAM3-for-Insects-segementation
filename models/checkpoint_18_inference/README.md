# The fine-tuned checkpoint

The weights are **not** stored in this repository — they are 3.14 GB. This directory holds the
checksum and the two scripts that fetch and verify them.

## Get the weights

```sh
./fetch_weights.sh          # downloads both parts from the GitHub Release, joins, verifies
```

```python
from sam3_insect import resolve_checkpoint
ckpt = resolve_checkpoint("release")   # or "local" with path=...
```

Either way you end up with `checkpoint_18_inference.pt`, 3,371,878,637 bytes, SHA-256

```
dd8a6ce0402a6c2d00b2849a3e08becc6f3aa4ececdc526580a54539c9c41829
```

## Sources

The [GitHub Release `checkpoint-18`](https://github.com/adambasha0/SAM3-for-Insects-segementation/releases/tag/checkpoint-18)
carries the checkpoint as **two ~1.57 GiB parts**, because GitHub caps a single file at 2 GB.

If you mirror the weights to a HuggingFace Hub repo — one file, resumable, convenient if you pull
them often — point the resolver at it:

```python
resolve_checkpoint("hf", hf_repo="you/your-repo", hf_filename="checkpoint_18_inference.pt")
```

## Joining parts by hand

Nothing here is a custom format — the parts are a plain byte-wise split, so you never need this
repository's tooling to recover your weights:

```sh
cat checkpoint_18_inference.pt.part-* > checkpoint_18_inference.pt
sha256sum -c checkpoint_18_inference.pt.sha256
```

`restore.sh` does exactly that with the checksum check wired in, for the case where you already have
both parts locally.

## Files

| file | what it is |
|---|---|
| `fetch_weights.sh` | Download from the release, join, verify, clean up the parts |
| `restore.sh` | Join parts you already have, and verify |
| `checkpoint_18_inference.pt.sha256` | Checksum of the reassembled checkpoint |

See [`../../MODEL_CARD.md`](../../MODEL_CARD.md) for what the checkpoint contains, how its fidelity
was verified, and why it ships as fp32 rather than fp16.
