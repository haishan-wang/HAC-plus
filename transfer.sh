#!/bin/bash
set -euo pipefail

# Sync scripts from triton to the local HAC-plus project directory.
REMOTE_HOST="triton"
REMOTE_BASE="/scratch/cs/gnnflows/Projects/AGSC/HAC-plus"
LOCAL_DIR="$HOME/Documents/Projects/AGSC/HAC-plus"

rsync -avz --progress "${REMOTE_HOST}:${REMOTE_BASE}/scripts" "${LOCAL_DIR}/"
rsync -avz --progress "${REMOTE_HOST}:${REMOTE_BASE}/outputs" "${LOCAL_DIR}/"
