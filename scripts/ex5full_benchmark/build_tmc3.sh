#!/bin/bash
# Build the MPEG G-PCC codec (tmc3) and put it on the PATH of HACplus_env (symlink into the env's bin/).
# HAC++ calls `tmc3` (utils/gpcc_utils.py) to entropy-code the anchor positions in the final encoding;
# see "Install tmc3" in the upstream README.
#   triton:  bash scripts/ex5full_benchmark/build_tmc3.sh            (login node is fine)
#   local:   bash scripts/ex5full_benchmark/build_tmc3.sh ~/Documents/Projects/AGSC/mpeg-pcc-tmc13
set -euo pipefail

TMC_ROOT=${1:-/scratch/cs/gnnflows/Projects/AGSC/mpeg-pcc-tmc13}
BUILD_DIR=${2:-$TMC_ROOT/build-release}

if command -v module > /dev/null 2>&1; then  # triton: cmake / g++ come from modules
    module load triton/2025.1-gcc
    module load cmake/3.30.5 gcc/13.3.0
    module load mamba
fi

cmake -S "$TMC_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target tmc3 -j 8
TMC3_BIN="$BUILD_DIR/tmc3/tmc3"
ls -l "$TMC3_BIN"

set +u  # conda activate.d scripts reference unset variables
eval "$(conda shell.bash hook)"
conda activate HACplus_env
set -u
ln -sf "$TMC3_BIN" "$CONDA_PREFIX/bin/tmc3"
echo "tmc3 -> $(command -v tmc3)"
