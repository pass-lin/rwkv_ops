"""
TensorFlow 版 generalized_delta_rule
前向用 tf.py_function 调 JAX CUDA 内核，反向同样走 JAX。
可 @tf.function 编译，可 tf.GradientTape 训练。
"""

import tensorflow as tf
from typing import Optional, Tuple
import jax.numpy as jnp
from .jax_cuda_kernel.wkv7_jax import get_jax_generalized_delta_rule


def transpose_head(x, head_first: bool):
    """(B, T, H, K) <-> (B, H, T, K)"""
    x = tf.cast(x, dtype=tf.float32)
    if head_first:
        return tf.transpose(x, (0, 2, 1, 3))
    return x


def get_tf_generalized_delta_rule(HEAD_SIZE=64):
    _, _wkv7_kernel, _wkv7_bwd_kernel = get_jax_generalized_delta_rule(HEAD_SIZE)

    # ---------- 底层 kernel 包装 ----------
    @tf.py_function(Tout=[tf.bfloat16, tf.float32, tf.float32])
    def _tf_wkv7_fwd(w, q, k, v, a, b, h0):
        """tf.py_function 包装 JAX 前向"""
        y, s, sa = _wkv7_kernel(
            jnp.asarray(w, jnp.bfloat16),
            jnp.asarray(q, jnp.bfloat16),
            jnp.asarray(k, jnp.bfloat16),
            jnp.asarray(v, jnp.bfloat16),
            jnp.asarray(a, jnp.bfloat16),
            jnp.asarray(b, jnp.bfloat16),
            jnp.asarray(h0, jnp.float32),
        )
        return (
            tf.convert_to_tensor(y, tf.bfloat16),
            tf.convert_to_tensor(s, tf.float32),
            tf.convert_to_tensor(sa, tf.float32),
        )

    @tf.py_function(Tout=[tf.bfloat16] * 6 + [tf.float32])
    def _tf_wkv7_bwd(w, q, k, v, a, b, dy, s, sa, dht):
        """tf.py_function 包装 JAX 反向"""
        dw, dq, dk, dv, da, db, dh0 = _wkv7_bwd_kernel(
            jnp.asarray(w, jnp.bfloat16),
            jnp.asarray(q, jnp.bfloat16),
            jnp.asarray(k, jnp.bfloat16),
            jnp.asarray(v, jnp.bfloat16),
            jnp.asarray(a, jnp.bfloat16),
            jnp.asarray(b, jnp.bfloat16),
            jnp.asarray(dy, jnp.bfloat16),
            jnp.asarray(s, jnp.float32),
            jnp.asarray(sa, jnp.float32),
            jnp.asarray(dht, jnp.bfloat16),
        )
        return tuple(
            tf.convert_to_tensor(g, dtype)
            for g, dtype in zip((dw, dq, dk, dv, da, db), [tf.bfloat16] * 6)
        ) + (tf.convert_to_tensor(dh0, tf.float32),)

    # ---------- 带梯度的前向 ----------
    @tf.custom_gradient
    def _wk7_tf(w, q, k, v, a, b, h0):
        y, s, sa = _tf_wkv7_fwd(w, q, k, v, a, b, h0)

        def grad(dy, dht):
            # dy 上层传来的 loss 对 y 的梯度
            # dht 对最后状态的梯度（没有就传 0）
            if dht is None:
                dht = tf.zeros_like(h0)
            grads = _tf_wkv7_bwd(w, q, k, v, a, b, dy, s, sa, dht)
            return grads  # (dw, dq, dk, dv, da, db, dh0)

        final_state = s[:, :, -1]  # (B, H, K, K)
        final_state = tf.transpose(final_state, [0, 1, 3, 2])  # 与 JAX 对齐
        return (y, final_state), grad

    # ---------- 用户接口 ----------
    def generalized_delta_rule(
        r: tf.Tensor,  # (B, T, H, K) 或 (B, H, T, K)
        w: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        a: tf.Tensor,
        b: tf.Tensor,
        initial_state: Optional[tf.Tensor] = None,
        output_final_state: bool = True,
        head_first: bool = False,
        chunk_len: int = 16,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        与 JAX 版接口 1:1 对齐，返回 (out, last_state)
        可 @tf.function  compile，可 tf.GradientTape 训练
        """
        dtype = r.dtype

        r = transpose_head(r, head_first)
        w = transpose_head(w, head_first)
        k = transpose_head(k, head_first)
        v = transpose_head(v, head_first)
        a = transpose_head(a, head_first)
        b = transpose_head(b, head_first)

        B, T, H, K = tf.unstack(tf.shape(r), num=4)
        if T % chunk_len != 0:
            raise ValueError(f"T={T} must be divisible by chunk_len={chunk_len}")

        if initial_state is None:
            h0 = tf.zeros([B, H, K, K], dtype=tf.float32)
        else:
            h0 = tf.cast(initial_state, tf.float32)

        # 带梯度前向
        out, last_state = _wk7_tf(w, r, k, v, a, b, h0)

        # 转回用户期望 dtype
        out = tf.cast(out, dtype)

        return (out, last_state) if output_final_state else out

    return generalized_delta_rule, _tf_wkv7_fwd, _tf_wkv7_bwd
