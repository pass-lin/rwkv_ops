from setuptools import setup, find_packages

setup(
    name="rwkv_ops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["keras"],  # 添加依赖项
)