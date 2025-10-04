# -*- coding: utf-8 -*-
"""
Rwkv Ops – multi-backend kernel library
Build:  python -m build
Upload: twine upload dist/*
"""

from pathlib import Path
from setuptools import setup, find_packages

PKG_NAME = "rwkv_ops"

KERNEL_PATTERNS = ("*.cpp", "*.cu", "*.h", "*.hpp", "*.cuh")


def gather_kernel_data():
    """返回 {package_name: [file_relative_paths]}"""
    pkg_dir = Path(PKG_NAME)
    files = []
    for pat in KERNEL_PATTERNS:
        files.extend(pkg_dir.rglob(pat))
    # 转成 setuptools 需要的 “相对包路径” 写法
    return [str(p.relative_to(pkg_dir).as_posix()) for p in files]


setup(
    name=PKG_NAME,
    version="0.2.3",
    packages=find_packages(),  # Python 包
    package_data={PKG_NAME: gather_kernel_data()},
    include_package_data=True,  # 强制打包 package_data
    python_requires=">=3.9",
    install_requires=["keras"],
    license="Apache-2.0",
    description="RWKV operator implementations for multiple backends",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/pass-lin/rwkv_ops",
    keywords="rwkv multi-backend kernel cuda jax torch",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
