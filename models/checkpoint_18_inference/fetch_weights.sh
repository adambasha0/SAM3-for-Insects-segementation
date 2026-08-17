#!/usr/bin/env bash
# Download the fine-tuned checkpoint from the GitHub Release, reassemble it and
# verify the SHA-256 -- no Python, no Git LFS.
#
#   ./fetch_weights.sh [OUTPUT_PATH]
#
# Default output: ./checkpoint_18_inference.pt  (3,371,878,637 bytes)
set -euo pipefail

cd "$(dirname "$0")"

REPO="${SAM3_INSECT_REPO:-adambasha0/SAM3-for-Insects-segementation}"
TAG="${SAM3_INSECT_TAG:-checkpoint-18}"
OUT="${1:-checkpoint_18_inference.pt}"
PARTS=(checkpoint_18_inference.pt.part-00 checkpoint_18_inference.pt.part-01)
BASE="https://github.com/${REPO}/releases/download/${TAG}"

if [ -s "${OUT}" ] && [ "$(stat --printf='%s' "${OUT}")" = "3371878637" ]; then
    echo "Already present: ${OUT}"
    exit 0
fi

for part in "${PARTS[@]}"; do
    if [ -s "${part}" ]; then
        echo "Have ${part}, skipping download."
        continue
    fi
    echo "Downloading ${part} ..."
    # -C - resumes a partial file, which matters for 1.57 GiB over a flaky link.
    if command -v curl >/dev/null 2>&1; then
        curl -fL -C - -o "${part}" "${BASE}/${part}"
    else
        wget -c -O "${part}" "${BASE}/${part}"
    fi
done

echo "Reassembling -> ${OUT}"
cat "${PARTS[@]}" > "${OUT}"

echo "Verifying sha256 ..."
EXPECTED="$(cut -d' ' -f1 < checkpoint_18_inference.pt.sha256)"
ACTUAL="$(sha256sum "${OUT}" | cut -d' ' -f1)"

if [ "${EXPECTED}" != "${ACTUAL}" ]; then
    echo "ERROR: checksum mismatch — the download is corrupt." >&2
    echo "  expected ${EXPECTED}" >&2
    echo "  actual   ${ACTUAL}" >&2
    rm -f "${OUT}"
    exit 1
fi

rm -f "${PARTS[@]}"
echo "OK: ${OUT} verified ($(stat --printf='%s' "${OUT}") bytes)."
echo
echo "Use it with:"
echo "  sam3_insect_predict -i IMAGE_OR_DIR -o out/ -w $(pwd)/${OUT}"
