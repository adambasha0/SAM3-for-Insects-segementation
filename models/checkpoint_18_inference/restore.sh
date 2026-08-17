#!/usr/bin/env bash
# Reassemble the split inference checkpoint and verify it is byte-identical.
#
#   ./restore.sh [OUTPUT_PATH]
#
# Default output: ./checkpoint_18_inference.pt
set -euo pipefail

cd "$(dirname "$0")"
OUT="${1:-checkpoint_18_inference.pt}"

if [ ! -s checkpoint_18_inference.pt.part-00 ] || \
   grep -qs 'git-lfs.github.com' checkpoint_18_inference.pt.part-00; then
    echo "ERROR: the part files are still Git LFS pointers, not real data." >&2
    echo "Run:  git lfs install && git lfs pull" >&2
    exit 1
fi

echo "Reassembling -> ${OUT}"
cat checkpoint_18_inference.pt.part-* > "${OUT}"

echo "Verifying sha256..."
EXPECTED="$(cat checkpoint_18_inference.pt.sha256)"
ACTUAL="$(sha256sum "${OUT}" | awk '{print $1}')"

if [ "${EXPECTED}" != "${ACTUAL}" ]; then
    echo "ERROR: checksum mismatch." >&2
    echo "  expected ${EXPECTED}" >&2
    echo "  actual   ${ACTUAL}" >&2
    exit 1
fi

echo "OK: ${OUT} verified byte-identical ($(stat --printf='%s' "${OUT}") bytes)."
