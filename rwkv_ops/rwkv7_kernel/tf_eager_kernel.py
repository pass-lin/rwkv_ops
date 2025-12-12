"""
TensorFlow 版 generalized_delta_rule
前向用 tf.py_function 调 JAX CUDA 内核，反向同样走 JAX。
可 @tf.function 编译，可 tf.GradientTape 训练。
"""

import tensorflow as tf
from typing import Optional, Tuple
import jax.numpy as jnp
from .jax_cuda_kernel.wkv7_jax import get_jax_generalized_delta_rule
from .jax_cuda_kernel_single.wkv7_single_step_jax import (
    get_jax_generalized_delta_rule_single_step,
)


def transpose_head(x, head_first: bool):
    """(B, T, H, K) <-> (B, H, T, K)"""
    x = tf.cast(x, dtype=tf.float32)
    if head_first:
        return tf.transpose(x, (0, 2, 1, 3))
    return x


def get_tf_generalized_delta_rule(HEAD_SIZE=64):
    generalized_delta_rule_inference = get_jax_generalized_delta_rule(HEAD_SIZE)[1]

    # ---------- 底层 kernel 包装 ----------
    @tf.py_function(Tout=[tf.bfloat16, tf.float32])
    def _tf_wkv7_fwd(w, q, k, v, a, b, h0):
        """tf.py_function 包装 JAX 前向"""
        y, s = generalized_delta_rule_inference(
            w=jnp.asarray(w, jnp.bfloat16),
            r=jnp.asarray(q, jnp.bfloat16),
            k=jnp.asarray(k, jnp.bfloat16),
            v=jnp.asarray(v, jnp.bfloat16),
            a=jnp.asarray(a, jnp.bfloat16),
            b=jnp.asarray(b, jnp.bfloat16),
            initial_state=jnp.asarray(h0, jnp.float32),
        )
        return (
            tf.convert_to_tensor(y, tf.bfloat16),
            tf.convert_to_tensor(s, tf.float32),
        )

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
        out, last_state = _tf_wkv7_fwd(w, r, k, v, a, b, h0)

        # 转回用户期望 dtype
        out = tf.cast(out, dtype)

        return (out, last_state) if output_final_state else out

    return generalized_delta_rule


def get_tf_generalized_delta_rule_single_step(HEAD_SIZE=64):
    # 获取 JAX 版本的单步 generalized delta rule
    _wkv7_single_step_kernel = get_jax_generalized_delta_rule_single_step(HEAD_SIZE)

    # ---------- 底层 kernel 包装 ----------
    @tf.py_function(Tout=[tf.bfloat16, tf.float32])
    def _tf_wkv7_single_step_fwd(w, r, k, v, a, b, h0):
        """tf.py_function 包装 JAX 单步前向"""
        y, s = _wkv7_single_step_kernel(
            w=jnp.asarray(w, jnp.bfloat16),
            r=jnp.asarray(r, jnp.bfloat16),
            k=jnp.asarray(k, jnp.bfloat16),
            v=jnp.asarray(v, jnp.bfloat16),
            a=jnp.asarray(a, jnp.bfloat16),
            b=jnp.asarray(b, jnp.bfloat16),
            initial_state=jnp.asarray(h0, jnp.float32),
        )
        return (
            tf.convert_to_tensor(y, tf.bfloat16),
            tf.convert_to_tensor(s, tf.float32),
        )

    # ---------- 用户接口 ----------
    def generalized_delta_rule_single_step(
        r: tf.Tensor,  # (B, 1, H, K) 或 (B, H, 1, K)
        w: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        a: tf.Tensor,
        b: tf.Tensor,
        initial_state: Optional[tf.Tensor] = None,
        output_final_state: bool = True,
        head_first: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        单步 generalized delta rule 实现
        与 JAX 版单步接口对齐，返回 (out, last_state)
        """
        dtype = r.dtype

        r = transpose_head(r, head_first)
        w = transpose_head(w, head_first)
        k = transpose_head(k, head_first)
        v = transpose_head(v, head_first)
        a = transpose_head(a, head_first)
        b = transpose_head(b, head_first)

        B, T, H, K = tf.unstack(tf.shape(r), num=4)
        if T != 1:
            raise ValueError(f"Single-step kernel requires T=1, but got T={T}")

        if initial_state is None:
            h0 = tf.zeros([B, H, K, K], dtype=tf.float32)
        else:
            h0 = tf.cast(initial_state, tf.float32)

        # 前向计算
        y, s = _tf_wkv7_single_step_fwd(w, r, k, v, a, b, h0)

        # 转回用户期望 dtype
        out = tf.cast(y, dtype)

        return (out, s) if output_final_state else out

    return generalized_delta_rule_single_step
