import argparse
import os
import numpy as np


def run_test(backend: str, kernel_type: str):
    print(f"\nTesting with KERAS_BACKEND={backend}, KERNEL_TYPE={kernel_type}")

    # 设置环境变量
    os.environ["KERAS_BACKEND"] = backend
    os.environ["KERNEL_TYPE"] = kernel_type
    from keras import ops

    from rwkv_ops import rwkv7_op

    B, T, H, C = 2, 16, 6, 64
    r = ops.array(np.random.randn(B, T, H, C), dtype="bfloat16")
    w = ops.array(np.random.randn(B, T, H, C), dtype="bfloat16")
    k = ops.array(np.random.randn(B, T, H, C), dtype="bfloat16")
    v = ops.array(np.random.randn(B, T, H, C), dtype="bfloat16")
    a = ops.array(np.random.randn(B, T, H, C), dtype="bfloat16")
    b = ops.array(np.random.randn(B, T, H, C), dtype="bfloat16")

    try:
        output, state = rwkv7_op(r, w, k, v, a, b,head_first=False)
        print("Output shape:", output.shape)
        if kernel_type == "cuda":
            assert output.shape == (B, T, H * C), (
                f"Expected output shape {(B, T, H * C)}, got {output.shape}"
            )
        else:
            assert output.shape == (B, T, H, C), (
                f"Expected output shape {(B, T, H, C)}, got {output.shape}"
            )
        print(
            "✅ Test passed at %s Backend and %s impplementation"
            % (backend, kernel_type)
        )
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test RWKV7 Op with different backends and kernel types."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "jax", "numpy", "tensorflow","openvino"],
        required=True,
    )
    parser.add_argument(
        "--kernel-type",
        type=str,
        choices=["triton", "cuda", "native"],
        required=True,
        help="Kernel type to use (triton, cuda, native)",
    )

    args = parser.parse_args()

    run_test(args.backend, args.kernel_type)
