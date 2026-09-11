"""DeltaNet chunkwise SANE Triton kernel 子模块入口。

未改动的 kernel 直接从 delta_net_chunk.triton 复用；
只在本目录覆盖需要加入 SANE 的 chunk_h 与 chunk_bwd_dhu。
"""

from ...delta_net_chunk.triton import (
    delta_net_chunk_bwd_dqk,
    delta_net_chunk_bwd_dv_local,
    delta_net_chunk_fwd_intra,
    delta_net_chunk_fwd_o,
    delta_net_chunk_l2norm_bwd,
    delta_net_chunk_l2norm_fwd,
    delta_net_chunk_prepare_wy_repr_bwd,
    delta_net_chunk_recompute_w_u,
)
from .chunk_bwd_dhu import delta_net_chunk_bwd_dhu_sane as delta_net_chunk_bwd_dhu
from .chunk_h import delta_net_chunk_fwd_h_sane as delta_net_chunk_fwd_h

__all__ = [
    "delta_net_chunk_bwd_dhu",
    "delta_net_chunk_bwd_dqk",
    "delta_net_chunk_bwd_dv_local",
    "delta_net_chunk_fwd_h",
    "delta_net_chunk_fwd_intra",
    "delta_net_chunk_fwd_o",
    "delta_net_chunk_l2norm_bwd",
    "delta_net_chunk_l2norm_fwd",
    "delta_net_chunk_prepare_wy_repr_bwd",
    "delta_net_chunk_recompute_w_u",
]
