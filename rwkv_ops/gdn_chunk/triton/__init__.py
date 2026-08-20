"""Gated DeltaNet chunkwise Triton kernel 子模块入口。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""

from .chunk_h import gdn_chunk_fwd_h as gdn_chunk_fwd_h
from .chunk_o import gdn_chunk_fwd_o as gdn_chunk_fwd_o
from .intra import gdn_chunk_fwd_intra as gdn_chunk_fwd_intra
from .l2norm import gdn_chunk_l2norm_fwd as gdn_chunk_l2norm_fwd
from .utils import chunk_local_cumsum as chunk_local_cumsum
from .wy import gdn_chunk_recompute_w_u as gdn_chunk_recompute_w_u
