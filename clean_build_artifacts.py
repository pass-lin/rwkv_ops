#!/usr/bin/env python3
"""
清理 RWKV-OPS 各 CUDA/FFI 算子产生的构建产物。

包括：
- rwkv6/rwkv7 的 JAX FFI build 目录与 .so
- rwkv7 单步 kernel 的 build 目录与 .so
- 项目根目录及算子目录下的 .ninja_log / .ninja_deps
- 各目录下的 __pycache__

用法：
    python clean_build_artifacts.py
"""

import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.resolve()

_CLEAN_PATTERNS = [
    # RWKV-6 JAX FFI
    "rwkv_ops/rwkv6_kernel/jax_cuda_kernel/build_*",
    "rwkv_ops/rwkv6_kernel/jax_cuda_kernel/*.so",
    # RWKV-7 JAX FFI
    "rwkv_ops/rwkv7_kernel/jax_cuda_kernel/build_*",
    "rwkv_ops/rwkv7_kernel/jax_cuda_kernel/*.so",
    # RWKV-7 single-step JAX FFI
    "rwkv_ops/rwkv7_kernel/jax_cuda_kernel_single/build_*",
    "rwkv_ops/rwkv7_kernel/jax_cuda_kernel_single/*.so",
    # RWKV-7-sn single-step JAX FFI
    "rwkv_ops/rwkv7_sn_kernel/jax_cuda_kernel/build_*",
    "rwkv_ops/rwkv7_sn_kernel/jax_cuda_kernel/*.so",
    # RWKV-7-sn single-step JAX FFI
    "rwkv_ops/rwkv7_sn_kernel/jax_cuda_kernel_single/build_*",
    "rwkv_ops/rwkv7_sn_kernel/jax_cuda_kernel_single/*.so",
    # 根目录 ninja 日志
    ".ninja_log",
    ".ninja_deps",
]


def _glob_paths(pattern: str):
    """支持 `*` 和 `**` 的简单 glob，返回绝对路径列表。"""
    base = _PROJECT_ROOT
    parts = pattern.replace("\\", "/").split("/")
    current = [base]

    for part in parts:
        next_current = []
        for c in current:
            if not c.exists():
                continue
            if part == "**":
                # 递归所有子目录
                next_current.extend(c.rglob("*"))
            elif "*" in part:
                next_current.extend(c.glob(part))
            else:
                p = c / part
                if p.exists():
                    next_current.append(p)
        current = next_current

    return sorted({p.resolve() for p in current})


def _remove(path: Path):
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        print(f"[DIR]  {path}")
    elif path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)
        print(f"[FILE] {path}")


def _clean_pycache():
    count = 0
    for pycache in _PROJECT_ROOT.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache, ignore_errors=True)
            print(f"[DIR]  {pycache}")
            count += 1
    return count


def clean_all():
    removed = 0
    for pattern in _CLEAN_PATTERNS:
        for path in _glob_paths(pattern):
            _remove(path)
            removed += 1

    removed += _clean_pycache()

    if removed == 0:
        print("没有找到需要清理的构建产物。")
    else:
        print(f"\n共清理 {removed} 个文件/目录。")
    return removed


def main():
    clean_all()


if __name__ == "__main__":
    main()
