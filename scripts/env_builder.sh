#!/bin/bash
set -e

# cd "$(dirname "$0")/.."

ENV_NAME=$(grep '^name:' environment.yml | awk '{print $2}')

# 1. Extract submodule sources (skip any that are already extracted).
cd submodules
for pkg in diff-gaussian-rasterization gridencoder simple-knn arithmetic; do
    if [ ! -d "$pkg" ] || [ -z "$(ls -A "$pkg" 2>/dev/null)" ]; then
        unzip -q -o "${pkg}.zip"
    fi
done
rm -rf __MACOSX
cd ..

# gridencoder hardcodes -std=c++14, but pytorch>=2.1 headers require C++17.
sed -i 's/std=c++14/std=c++17/g' submodules/gridencoder/setup.py

# 2. Create (or update) the conda env with the conda deps + plain pip deps.
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "Conda env '${ENV_NAME}' already exists, updating it."
    conda env update --file environment.yml
else
    conda env create --file environment.yml
fi

# 3. Build and install the CUDA-extension submodules. These setup.py files
# `import torch` at build time, so they must be installed with
# --no-build-isolation against an env that already has torch installed
# (pip's isolated build env otherwise can't see it).
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
pip install --no-build-isolation \
    submodules/diff-gaussian-rasterization \
    submodules/gridencoder \
    submodules/arithmetic

echo "Environment '${ENV_NAME}' is ready. Activate it with: conda activate ${ENV_NAME}"
