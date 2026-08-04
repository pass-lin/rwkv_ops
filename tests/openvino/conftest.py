"""OpenVINO 后端测试的 session 级配置。"""

import os

# 必须在 import keras / rwkv_ops 之前设定 KERAS_BACKEND。
os.environ.setdefault("KERAS_BACKEND", "openvino")

import pytest  # noqa: E402

pytest.importorskip("openvino")
pytest.importorskip("keras")
