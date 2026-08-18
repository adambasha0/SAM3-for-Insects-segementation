#!/usr/bin/env python3
"""Create (or update) the HuggingFace Hub mirror of the checkpoint.

    export HF_TOKEN=hf_...        # a write token, see the scoping note below
    python tools/publish_to_hf.py path/to/checkpoint_18_inference.pt \
        --repo USER/sam3-for-insects-segmentation

The Hub is the better of the two sources for a repeat user: one resumable file
instead of two parts to join, and no 2 GB per-file limit. The GitHub Release
stays the default because it needs no account at all.

**Token scoping is the thing that bites.** A fine-grained token is bound to the
namespace that issued it. If the token belongs to user `alice`, it can only write
to `alice/...` — creating `bob/...` fails even when you can see that repo in a
browser. Check before uploading 3 GB:

    python tools/publish_to_hf.py --whoami
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_CARD = """---
license: other
license_name: sam-license
license_link: https://github.com/adambasha0/SAM3-for-Insects-segementation/blob/main/LICENSE
pipeline_tag: mask-generation
library_name: sam3-insect
tags:
  - instance-segmentation
  - object-detection
  - insects
  - arthropods
  - entomology
  - sam3
---

# SAM 3 for Insects

See the [repository](https://github.com/adambasha0/SAM3-for-Insects-segementation)
for the code, the CLI and the Colab notebook.
"""


def resolve_token(explicit: str | None) -> str:
    token = explicit or os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN")
    if not token:
        sys.exit("No token. Set HF_TOKEN to a write token, or pass --token.")
    return token


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", nargs="?", help="Checkpoint file to upload")
    parser.add_argument("--repo", default=None, help="e.g. user/sam3-for-insects-segmentation")
    parser.add_argument(
        "--filename",
        default="checkpoint_18_inference.pt",
        help="Name the file gets in the repo",
    )
    parser.add_argument("--card", default=None, help="Markdown file to upload as README.md")
    parser.add_argument("--private", action="store_true", help="Create the repo private")
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--whoami",
        action="store_true",
        help="Print the token's identity and write scope, then exit",
    )
    parser.add_argument(
        "--card-only", action="store_true", help="Update README.md and skip the weights"
    )
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi

    token = resolve_token(args.token)
    api = HfApi(token=token)

    identity = api.whoami()
    access = identity.get("auth", {}).get("accessToken", {})
    scoped = access.get("fineGrained", {}).get("scoped", [])
    namespaces = [s["entity"]["name"] for s in scoped] or [identity["name"]]
    print(f"token: {access.get('displayName') or '(classic)'}  role={access.get('role')}")
    print(f"identity: {identity['name']}   writable namespaces: {', '.join(namespaces)}")
    if args.whoami:
        return 0

    if not args.repo:
        sys.exit("--repo is required (e.g. --repo %s/sam3-for-insects-segmentation)"
                 % namespaces[0])
    owner = args.repo.split("/")[0]
    if owner not in namespaces:
        print(
            f"\nWARNING: the token is scoped to {namespaces} but you are writing to "
            f"'{owner}'. This will fail with a 403 rather than a helpful message.",
            file=sys.stderr,
        )

    print(f"\nCreating or updating {args.repo} (private={args.private}) ...")
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    card = Path(args.card).read_text() if args.card else DEFAULT_CARD
    print("Uploading README.md (the model card) ...")
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="Update model card",
    )

    if not args.card_only:
        if not args.checkpoint:
            sys.exit("pass a checkpoint path, or use --card-only")
        path = Path(args.checkpoint)
        print(f"Uploading {path.name} ({path.stat().st_size / 2**30:.2f} GiB) ...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=args.filename,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Add {args.filename}",
        )

    print("\nFiles in the repo:")
    for name in api.list_repo_files(args.repo):
        print("  ", name)
    print(f"\nhttps://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
