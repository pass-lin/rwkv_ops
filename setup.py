from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import os
import subprocess
from pathlib import Path

KERNEl_PATTERNS = ("*.cpp", "*.cu", "*.h", "*.hpp", "*.cuh", "*.hip", "*.cc", "*.txt")


def gather_kernel_data():
    """收集所有 CUDA 和 C++ 文件"""
    pkg_dir = Path("rwkv_ops")
    files = []
    for pat in KERNEl_PATTERNS:
        files.extend(pkg_dir.rglob(pat))
    return [str(p.relative_to(pkg_dir).as_posix()) for p in files]


class CMakeBuild(build_ext):
    """自定义构建扩展类"""

    def build_extension(self, ext):
        """构建扩展"""
        src_dir = os.path.abspath(os.path.dirname(__file__))
        build_dir = os.path.join(src_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        install_dir = os.path.join(src_dir, "rwkv_ops")

        cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", ".", "-j"], cwd=build_dir)
        subprocess.check_call(["cmake", "--install", "."], cwd=build_dir)


setup(
    name="rwkv_ops",
    version="0.3.0",
    packages=find_packages(),
    package_data={"rwkv_ops": gather_kernel_data()},
    include_package_data=True,
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.9",
    install_requires=[],
    license="Apache-2.0",
    description="RWKV operator implementations for multiple backends",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/pass-lin/rwkv_ops",
    keywords="rwkv multi-backend kernel cuda",
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
