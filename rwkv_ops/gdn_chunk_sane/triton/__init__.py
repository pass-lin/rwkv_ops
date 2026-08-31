"""Gated DeltaNet chunkwise SANE Triton kernel 子模块入口。

未改动的 kernel 直接从 gdn_chunk.triton 复用；
只在本目录覆盖需要加入 SANE 的 chunk_h 与 chunk_bwd_dhu。
"""

from ...gdn_chunk.triton import (
    chunk_local_cumsum,
    gdn_chunk_bwd_dqkwg,
    gdn_chunk_bwd_dv_local,
    gdn_chunk_fwd_intra,
    gdn_chunk_fwd_o,
    gdn_chunk_l2norm_bwd,
    gdn_chunk_l2norm_fwd,
    gdn_chunk_prepare_wy_repr_bwd,
    gdn_chunk_recompute_w_u,
)
from .chunk_h import gdn_chunk_fwd_h_sane as gdn_chunk_fwd_h
from .chunk_bwd_dhu import gdn_chunk_bwd_dhu_sane as gdn_chunk_bwd_dhu

__all__ = [
    "gdn_chunk_bwd_dhu",
    "gdn_chunk_bwd_dqkwg",
    "gdn_chunk_bwd_dv_local",
    "gdn_chunk_fwd_h",
    "gdn_chunk_fwd_intra",
    "gdn_chunk_fwd_o",
    "gdn_chunk_l2norm_bwd",
    "gdn_chunk_l2norm_fwd",
    "chunk_local_cumsum",
    "gdn_chunk_prepare_wy_repr_bwd",
    "gdn_chunk_recompute_w_u",
]
