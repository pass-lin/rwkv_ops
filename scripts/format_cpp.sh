#!/usr/bin/env bash
# 使用 LLVM style 格式化仓库内所有 .cu/.cuh/.cpp/.h 文件。
# 依赖：pip install clang-format

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v clang-format >/dev/null 2>&1; then
    echo "clang-format not found. Run: pip install clang-format"
    exit 1
fi

if [ ! -f .clang-format ]; then
    echo ".clang-format not found in repo root."
    exit 1
fi

mapfile -t FILES < <(find . -type f \( \
    -name '*.cu' -o \
    -name '*.cuh' -o \
    -name '*.cpp' -o \
    -name '*.h' \
\) \
    -not -path './.git/*' \
    -not -path './build*/*' \
    -not -path './dist/*' \
    -not -path './.pytest_cache/*' \
    -not -path './.ruff_cache/*' \
    -not -path './.vscode/*' \
    -not -path './__pycache__/*' \
    | sort)

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No C/CUDA files to format."
    exit 0
fi

echo "Formatting ${#FILES[@]} C/CUDA files with LLVM style..."
clang-format -style=file -i "${FILES[@]}"
echo "Done."
