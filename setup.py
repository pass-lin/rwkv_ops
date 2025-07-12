# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="rwkv_ops",
    version="0.1.1",
    packages=find_packages(),
    install_requires=["keras"],  # 添加依赖项
    license="Apache 2.0",  # 指定许可证类型
    long_description=open("README.md").read(),  # 从 README.md 文件中读取长描述
    long_description_content_type="text/markdown",  # 指定长描述的格式
    url="https://github.com/your-org/rwkv_ops",  # 项目主页
    keywords="rwkv attention cuda triton pytorch jax",  # 关键词
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
