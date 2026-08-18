#!/usr/bin/env python3
"""Create (or update) a GitHub Release and upload the weight parts to it.

    export GITHUB_TOKEN=ghp_...          # needs the "contents: write" scope
    python tools/publish_release.py dist/ --tag checkpoint-18

Release assets, not Git LFS, are the right home for a 3 GB checkpoint on a free
account: LFS on GitHub Free is capped at 10 GiB of *bandwidth* per month -- about
three downloads of this file -- whereas release assets are plain HTTPS with no
bandwidth accounting. They also skip the repository clone entirely, so
``git clone`` stays at a few megabytes.

Uses only the standard library, so it works without the ``gh`` CLI installed.
Re-running is safe: an asset already present at the right size is skipped, and a
size mismatch deletes the stale asset before re-uploading.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"


def resolve_token(explicit: str | None) -> str:
    """Token from --token, then GITHUB_TOKEN / GH_TOKEN."""
    token = explicit or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        sys.exit(
            "No token. Set GITHUB_TOKEN (a fine-grained PAT with 'contents: write' "
            "on the target repo) or pass --token."
        )
    return token


def make_api(token: str):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "sam3-insect-publish",
    }

    def call(url, data=None, method=None, extra=None, length=None):
        request = urllib.request.Request(url, data=data, method=method)
        for key, value in {**headers, **(extra or {})}.items():
            request.add_header(key, value)
        if length is not None:
            # urllib will not infer this from a file object, and GitHub needs it.
            request.add_header("Content-Length", str(length))
        try:
            with urllib.request.urlopen(request) as response:
                raw = response.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:1000]
            print(f"HTTP {exc.code} on {method or 'GET'} {url}\n{body}", file=sys.stderr)
            raise

    return call


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("assets", nargs="+", help="Files or a directory of files to upload")
    parser.add_argument("--repo", default="adambasha0/SAM3-for-Insects-segementation")
    parser.add_argument("--tag", default="checkpoint-18")
    parser.add_argument("--target", default="main", help="Commit-ish the tag points at")
    parser.add_argument("--title", default=None)
    parser.add_argument("--notes-file", default=None, help="Markdown file for the body")
    parser.add_argument("--token", default=None)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args(argv)

    call = make_api(resolve_token(args.token))

    paths: list[Path] = []
    for entry in args.assets:
        path = Path(entry)
        if path.is_dir():
            paths.extend(sorted(p for p in path.iterdir() if p.is_file()))
        elif path.is_file():
            paths.append(path)
        else:
            sys.exit(f"not found: {entry}")
    if not paths:
        sys.exit("nothing to upload")

    # --- find or create the release ---------------------------------------
    try:
        release = call(f"{API}/repos/{args.repo}/releases/tags/{args.tag}")
        print(f"Release {args.tag} exists (id {release['id']})")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        body = Path(args.notes_file).read_text() if args.notes_file else ""
        release = call(
            f"{API}/repos/{args.repo}/releases",
            data=json.dumps(
                {
                    "tag_name": args.tag,
                    "target_commitish": args.target,
                    "name": args.title or args.tag,
                    "body": body,
                    "draft": args.draft,
                    "prerelease": False,
                }
            ).encode(),
            method="POST",
            extra={"Content-Type": "application/json"},
        )
        print(f"Created release {args.tag} (id {release['id']})")

    release_id = release["id"]
    existing = {a["name"]: a for a in release.get("assets", [])}

    # --- upload -----------------------------------------------------------
    for path in paths:
        size = path.stat().st_size
        if size >= 2 * 2**30:
            print(f"  {path.name}: {size / 2**30:.2f} GiB is over GitHub's 2 GB "
                  "per-asset limit — split it first")
            return 1

        if path.name in existing:
            if existing[path.name]["size"] == size:
                print(f"  {path.name}: already uploaded, skipping")
                continue
            print(f"  {path.name}: size differs, replacing")
            call(
                f"{API}/repos/{args.repo}/releases/assets/{existing[path.name]['id']}",
                method="DELETE",
            )

        print(f"  {path.name}: uploading {size / 2**30:.2f} GiB ...", flush=True)
        with open(path, "rb") as fh:
            asset = call(
                f"{UPLOADS}/repos/{args.repo}/releases/{release_id}/assets"
                f"?name={path.name}",
                data=fh,
                method="POST",
                extra={"Content-Type": "application/octet-stream"},
                length=size,
            )
        print(f"    {asset['browser_download_url']}")

    final = call(f"{API}/repos/{args.repo}/releases/tags/{args.tag}")
    print(f"\n{final['html_url']}")
    for asset in final["assets"]:
        print(f"  {asset['name']}  {asset['size']} bytes  state={asset['state']}")

    print(
        "\nNote: if the browser_download_url 404s for an anonymous client, the "
        "repository is private — release assets inherit repository visibility."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
