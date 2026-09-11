"""DeltaNet chunkwise Triton kernel 子模块入口。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""

from .chunk_bwd_dhu import delta_net_chunk_bwd_dhu as delta_net_chunk_bwd_dhu
from .chunk_bwd_dqk import delta_net_chunk_bwd_dqk as delta_net_chunk_bwd_dqk
from .chunk_bwd_dv import delta_net_chunk_bwd_dv_local as delta_net_chunk_bwd_dv_local
from .chunk_h import delta_net_chunk_fwd_h as delta_net_chunk_fwd_h
from .chunk_o import delta_net_chunk_fwd_o as delta_net_chunk_fwd_o
from .intra import delta_net_chunk_fwd_intra as delta_net_chunk_fwd_intra
from .l2norm import delta_net_chunk_l2norm_bwd as delta_net_chunk_l2norm_bwd
from .l2norm import delta_net_chunk_l2norm_fwd as delta_net_chunk_l2norm_fwd
from .wy import delta_net_chunk_recompute_w_u as delta_net_chunk_recompute_w_u
from .wy_bwd import (
    delta_net_chunk_prepare_wy_repr_bwd as delta_net_chunk_prepare_wy_repr_bwd,
)

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
