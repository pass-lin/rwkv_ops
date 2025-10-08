#!/usr/bin/env bash
set -euo pipefail

############################
# 1. 本地只构建，不安装
############################
echo "==> 清理并构建"
rm -rf build dist *.egg-info
find . -name '*.so' -delete
find . -name build -type d -exec rm -rf {} + 2>/dev/null || true

export KERNEL_TYPE="native"
python -m build --wheel

############################
# 2. 去临时目录，远离源码
############################
TMPDIR=$(mktemp -d)
cd "$TMPDIR"

############################
# 3. 卸载旧包（保证卸的是 site-packages）
############################
pip uninstall -y rwkv_ops || true

############################
# 4. 安装新包（强制装到 site-packages）
############################
pip install --force-reinstall --no-deps "$OLDPWD"/dist/*.whl

############################
# 5. 验证：确认 import 到的不是本地源码
############################
INSTALLED=$(python3 -c 'import rwkv_ops, pathlib; print(pathlib.Path(rwkv_ops.__path__[0]).resolve())')
if [[ "$INSTALLED" == "$OLDPWD"* ]]; then
    echo "ERROR: 仍然 import 到本地源码目录，脚本中止！" >&2
    exit 1
fi
echo "==> 成功安装到 site-packages：$INSTALLED"

############################
# 6. 可选：彻底删除 site-packages 里的整个包（杀空）
############################
echo "==> 杀空 site-packages 里的 rwkv_ops"
rm -rf "$INSTALLED"

############################
# 7. 再装一次（真正干净）
############################
pip install --force-reinstall --no-deps "$OLDPWD"/dist/*.whl

############################
# 8. 收尾
############################
cd "$OLDPWD"
rm -rf "$TMPDIR"
echo "==> 全部完成，本地源码毫发无损"