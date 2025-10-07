find . -name build -type d -exec rm -rf {} +
find . -name '*.so' -delete
rm -rf rwkv_ops.egg-info
rm -rf dist
rm -rf build
export KERNEL_TYPE="native"
python -m build --wheel
#先装上
pip install dist/rwkv_ops-0.3.0-py3-none-any.whl
#!/usr/bin/env bash
set -euo pipefail

# 1. 拿到 rwkv_ops 包的真实路径（__path__[0]）
pkg_path=$(python3 -c 'import rwkv_ops, pathlib, sys; \
                       p=pathlib.Path(rwkv_ops.__path__[0]).resolve(); \
                       print(p)' )

if [[ -z $pkg_path ]]; then
    echo "ERROR: 无法获取 rwkv_ops 包路径" >&2
    exit 1
fi

# 2. 删掉编译好的部分
build_dir="$pkg_path/rwkv7_kernel/jax_cuda_kernel/build"
so_file="$pkg_path/rwkv7_kernel/jax_cuda_kernel/wkv7.so"

# 3. 删除（带提示，防止误删）
for target in "$build_dir" "$so_file"; do
    if [[ -e $target ]]; then
        echo "Removing  $target"
        rm -rf "$target"
    else
        echo "Skip      $target  (not exist)"
    fi
done

echo "Done."
pip uninstall rwkv_ops -y
#装上没有构建过的部分
pip install dist/rwkv_ops-0.3.0-py3-none-any.whl
