# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 delta attention (KDA) for Intel GPUs.

The NVIDIA and ROCm layers each drive three separate KDA kernels and do the
prefill/decode/spec-decode routing in Python. ``vllm-xpu-kernels`` instead
exposes one fused ``kda_attention`` op that performs the causal convolution and
the delta rule together and does that routing inside the kernel, so this layer
is essentially the projection maths plus a single op call.

Everything up to and including ``in_proj_qkvgfab`` matches ``amd/kda.py``; only
``_forward`` diverges.
"""

import torch
from einops import rearrange
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.distributed import divide
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention

# Generic KDA helpers, shared with Kimi-Linear. They are neither XPU- nor
# K3-specific, so they are imported rather than duplicated.
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    _KDA_GATE_LOGBOUND_MIN,
    _KimiGDNMergedColumnParallelLinear,
    _make_fused_conv1d_weight_loader,
    a_log_weight_loader,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig


def is_xpu_kda_supported() -> bool:
    """Whether the fused XPU KDA op is present in the installed kernels."""
    return hasattr(torch.ops, "_xpu_C") and hasattr(torch.ops._xpu_C, "kda_attention")


class KimiK3DeltaAttention(GatedDeltaNetAttention):
    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        if not is_xpu_kda_supported():
            raise RuntimeError(
                "The XPU Kimi-K3 KDA layer requires a vllm-xpu-kernels build "
                "that registers torch.ops._xpu_C.kda_attention."
            )

        kda_config = config.linear_attn_config  # type: ignore[attr-defined]
        assert kda_config is not None, "linear_attn_config must be set"
        assert kda_config.get("use_full_rank_gate", False), (
            "KimiK3DeltaAttention requires use_full_rank_gate; the low-rank "
            "gate path belongs to the shared Kimi-Linear layer."
        )

        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        self.projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(self.projection_size, self.tp_size)
        self.conv_size = kda_config["short_conv_kernel_size"]
        self.use_full_rank_gate = True

        # Keep f_a before the narrow beta shard, then pad each TP-local row to
        # select the aligned BF16 GEMM path.
        qkvg_output_sizes = [self.projection_size] * 4
        in_proj_output_sizes = qkvg_output_sizes + [
            self.head_dim,
            self.num_heads,
        ]
        local_output_size = (
            4 * self.local_projection_size + self.head_dim + self.local_num_heads
        )
        self.in_proj_padding = -local_output_size % 16
        if self.in_proj_padding:
            in_proj_output_sizes.append(self.in_proj_padding * self.tp_size)

        self.in_proj_qkvgfab = _KimiGDNMergedColumnParallelLinear(
            self.hidden_size,
            in_proj_output_sizes,
            replicated_shard_id=4,
            tp_size=self.tp_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvgfab",
        )
        if self.in_proj_padding:
            self.in_proj_qkvgfab.weight.data[-self.in_proj_padding :].zero_()

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32)
        )
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # One packed parameter and cache. The XPU op slices the weight into
        # Q/K/V itself and reads the packed conv cache in place, so unlike the
        # Triton layers nothing here has to be re-materialized per step.
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=3 * self.projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": _make_fused_conv1d_weight_loader(
                    [self.projection_size] * 3,
                    self.tp_size,
                    self.tp_rank,
                )
            },
        )

        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": a_log_weight_loader(0)})

        self.gate_lower_bound: float | None = kda_config.get("gate_lower_bound", None)
        if self.gate_lower_bound is not None:
            assert _KDA_GATE_LOGBOUND_MIN <= self.gate_lower_bound < 0, (
                "KDA gate lower bound must be in "
                f"[{_KDA_GATE_LOGBOUND_MIN}, 0). "
                f"Got {self.gate_lower_bound}."
            )
        self.use_safe_gate = self.gate_lower_bound is not None

        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            self.projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]

        split_sizes = [
            3 * self.local_projection_size,
            self.local_projection_size,
            self.head_dim,
            self.local_num_heads,
        ]
        if self.in_proj_padding:
            split_sizes.append(self.in_proj_padding)
        projected = projected_qkvgfab.split(split_sizes, dim=-1)
        mixed_qkv, g_proj_states, f_a, beta = projected[:4]

        g1 = self.f_b_proj(f_a)[0]
        # The kernel applies sigmoid to the raw logits itself but reads them as
        # float32, whereas the projection runs in the model dtype.
        beta = beta.unsqueeze(0).float()
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        self._forward(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            core_attn_out=core_attn_out,
        )
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        return self.o_proj(core_attn_out)[0]

    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        if get_forward_context().attn_metadata is None:
            return

        # `kda_attention` fuses the causal convolution with the delta rule and
        # picks the chunked, recurrent or spec-decode path per sub-batch from
        # the metadata, so there is no prefill/decode branch to take here.
        #
        # It also indexes Q/K/V by row stride, so these stay views into
        # `mixed_qkv`; copying them dense would cost a full pass over the
        # projection for no benefit.
        q, k, v = mixed_qkv.split(self.local_projection_size, dim=-1)
        torch.ops.vllm.kda_attention_core_xpu(
            core_attn_out,
            q,
            k,
            v,
            g1,
            beta,
            self.prefix,
        )
        core_attn_out.copy_(self.o_norm(core_attn_out, g2))
