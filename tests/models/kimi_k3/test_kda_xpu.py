# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Intel GPU (XPU) Kimi-K3 KDA path.

The XPU kernel fuses the causal convolution and the delta rule into a single
``torch.ops._xpu_C.kda_attention`` call and routes prefill / decode /
speculative-decode sub-batches internally, so these tests check it against a
float64 recurrent reference built from the same definition the Triton layers
use, rather than against another kernel.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="The fused KDA kernel is only built for Intel GPUs.",
)

DEVICE = "xpu"


def _kda_available() -> bool:
    return hasattr(torch.ops, "_xpu_C") and hasattr(torch.ops._xpu_C, "kda_attention")


requires_kda = pytest.mark.skipif(
    not _kda_available(),
    reason="vllm-xpu-kernels was built without torch.ops._xpu_C.kda_attention.",
)


def _log_gate(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Per-token log decay, mirroring ``kda_gate.hpp``.

    ``raw_gate`` is ``[T, H, D]``; ``a_log`` is ``[H]``; ``dt_bias`` is
    ``[H * D]``.
    """
    num_heads, head_dim = raw_gate.shape[1], raw_gate.shape[2]
    x = raw_gate + dt_bias.view(1, num_heads, head_dim)
    head_a = -torch.exp(a_log).view(1, num_heads, 1)
    if lower_bound is None:
        return head_a * torch.nn.functional.softplus(x)
    return lower_bound * torch.sigmoid(-head_a * x)


def _causal_conv1d_ref(
    x: torch.Tensor, weight: torch.Tensor, initial_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """SiLU causal depthwise conv over one sequence.

    ``x`` is ``[T, C]``, ``weight`` is ``[C, W]`` and ``initial_state`` is
    ``[C, W - 1]`` holding the last ``W - 1`` inputs of the previous step.
    """
    width = weight.shape[1]
    padded = torch.cat([initial_state, x.transpose(0, 1)], dim=-1)
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        window = padded[:, t : t + width]
        out[t] = (window * weight).sum(-1)
    new_state = padded[:, -(width - 1) :].clone()
    return torch.nn.functional.silu(out), new_state


def _kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recurrent KDA for one sequence, in float64.

    ``q``/``k``/``v`` are ``[T, H, D]`` post-convolution activations. ``state``
    is ``[H, V, K]`` -- value-major, matching the kernel's ``recurrent_state``
    cache layout, so the decay applies along the trailing key axis. Returns
    ``[T, H, D]`` outputs and the final state.
    """
    seq_len, num_heads, head_dim = q.shape
    decay = torch.exp(_log_gate(raw_gate, a_log, dt_bias, lower_bound))
    beta = torch.sigmoid(raw_beta)
    scale = head_dim**-0.5

    # The kernel adds eps to the sum of squares, not to the norm.
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * scale
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)

    out = torch.zeros_like(v)
    for t in range(seq_len):
        state = state * decay[t].unsqueeze(-2)
        kv = (state * k[t].unsqueeze(-2)).sum(-1)
        delta = (v[t] - kv) * beta[t].unsqueeze(-1)
        state = state + delta.unsqueeze(-1) * k[t].unsqueeze(-2)
        out[t] = (state * q[t].unsqueeze(-2)).sum(-1)
    return out, state


def _run_kda_attention(
    *,
    seq_lens: list[int],
    num_heads: int = 4,
    head_dim: int = 64,
    width: int = 4,
    lower_bound: float | None,
    num_prefills: int,
    num_decodes: int,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> None:
    """Build one batch, run the fused op and compare to the reference."""
    torch.manual_seed(seed)
    hidden_dim = num_heads * head_dim
    batch_size = len(seq_lens)
    num_actual_tokens = sum(seq_lens)

    # vLLM hands the kernel three row-strided views of one fused projection.
    fused = (torch.randn(num_actual_tokens, 3 * hidden_dim, device=DEVICE) * 0.3).to(
        dtype
    )
    q_proj, k_proj, v_proj = fused.split(hidden_dim, dim=-1)

    raw_gate = (
        torch.randn(1, num_actual_tokens, num_heads, head_dim, device=DEVICE) * 0.3
    ).to(dtype)
    raw_beta = torch.randn(1, num_actual_tokens, num_heads, device=DEVICE)
    weights = [
        torch.randn(hidden_dim, width, dtype=torch.float32, device=DEVICE) * 0.2
        for _ in range(3)
    ]
    # `head_a` is `-exp(a_log)`, so a_log near 0 decays the state by ~50% per
    # token. Over a 64-token chunk that spans ~45 log-units, and the chunked
    # backend folds exp(+/-G) into bf16 GEMM operands, so such a gate loses
    # precision there by construction. Trained KDA gates retain context far
    # longer; offsetting a_log keeps this test in that realistic regime.
    a_log = torch.randn(num_heads, device=DEVICE) * 0.2 - 2.0
    dt_bias = torch.randn(hidden_dim, device=DEVICE) * 0.2

    query_start_loc = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        device=DEVICE,
        dtype=torch.int32,
    )
    state_indices = torch.arange(batch_size, device=DEVICE, dtype=torch.int32)
    has_initial_state = torch.ones(batch_size, device=DEVICE, dtype=torch.bool)

    conv_state = torch.randn(batch_size, 3 * hidden_dim, width - 1, device=DEVICE) * 0.2
    recurrent_state = (
        torch.randn(batch_size, num_heads, head_dim, head_dim, device=DEVICE) * 0.1
    )
    conv_state_in = conv_state.clone()
    recurrent_state_in = recurrent_state.clone()

    output = torch.zeros(
        1, num_actual_tokens, num_heads, head_dim, dtype=dtype, device=DEVICE
    )
    torch.ops._xpu_C.kda_attention(
        output,
        q_proj,
        k_proj,
        v_proj,
        raw_gate,
        raw_beta,
        conv_state,
        recurrent_state,
        *weights,
        a_log,
        dt_bias,
        num_prefills,
        num_decodes,
        0,
        has_initial_state,
        query_start_loc,
        None,
        state_indices,
        None,
        None,
        None,
        None,
        num_actual_tokens,
        lower_bound,
    )

    # ---- reference ----
    ref_out = torch.zeros(num_actual_tokens, num_heads, head_dim, dtype=torch.float64)
    ref_conv_state = conv_state_in.double().cpu()
    ref_recurrent_state = recurrent_state_in.double().cpu()
    fused64 = fused.double().cpu()
    raw_gate64 = raw_gate[0].double().cpu()
    raw_beta64 = raw_beta[0].double().cpu()
    weights64 = [w.double().cpu() for w in weights]
    a_log64, dt_bias64 = a_log.double().cpu(), dt_bias.double().cpu()

    start = 0
    for seq in range(batch_size):
        end = start + seq_lens[seq]
        conv_out = []
        for idx in range(3):
            channels = slice(idx * hidden_dim, (idx + 1) * hidden_dim)
            dense, new_state = _causal_conv1d_ref(
                fused64[start:end, channels],
                weights64[idx],
                ref_conv_state[seq, channels],
            )
            ref_conv_state[seq, channels] = new_state
            conv_out.append(dense.view(-1, num_heads, head_dim))
        out_seq, final_state = _kda_reference(
            *conv_out,
            raw_gate64[start:end],
            raw_beta64[start:end],
            a_log64,
            dt_bias64,
            ref_recurrent_state[seq],
            lower_bound,
        )
        ref_out[start:end] = out_seq
        ref_recurrent_state[seq] = final_state
        start = end

    # bf16 activations bound the achievable accuracy; the measured worst case
    # over these shapes and both backends is ~9e-4.
    torch.testing.assert_close(output[0].double().cpu(), ref_out, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(
        conv_state.double().cpu(), ref_conv_state, atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(
        recurrent_state.double().cpu(),
        ref_recurrent_state,
        atol=5e-3,
        rtol=5e-3,
    )


@requires_kda
@pytest.mark.parametrize("seq_len", [1, 7, 64, 129, 512])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
@torch.inference_mode()
def test_kda_attention_prefill_matches_reference(seq_len, lower_bound):
    """Prefill of a single sequence, spanning the chunked/recurrent cutoff."""
    _run_kda_attention(
        seq_lens=[seq_len],
        lower_bound=lower_bound,
        num_prefills=1,
        num_decodes=0,
        seed=seq_len,
    )


@requires_kda
@pytest.mark.parametrize("batch_size", [1, 4, 17])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
@torch.inference_mode()
def test_kda_attention_decode_matches_reference(batch_size, lower_bound):
    """Pure-decode batch: every sequence contributes exactly one token."""
    _run_kda_attention(
        seq_lens=[1] * batch_size,
        lower_bound=lower_bound,
        num_prefills=0,
        num_decodes=batch_size,
        seed=batch_size,
    )


@requires_kda
@pytest.mark.parametrize("lower_bound", [None, -5.0])
@torch.inference_mode()
def test_kda_attention_mixed_batch_matches_reference(lower_bound):
    """Decode-first mixed batch, the layout the GDN metadata builder emits.

    The Triton layers have to split this batch because their chunk kernel
    returns NaN for length-1 sequences; the XPU op handles it in one call, so
    this is the case most likely to regress if that routing changes.
    """
    _run_kda_attention(
        seq_lens=[1, 1, 1, 96, 257],
        lower_bound=lower_bound,
        num_prefills=2,
        num_decodes=3,
        seed=5,
    )


@requires_kda
@torch.inference_mode()
def test_kda_attention_accepts_row_strided_projections():
    """q/k/v stay views into the fused projection; a dense copy must not matter.

    This is what lets the layer skip a full pass over ``mixed_qkv``.
    """
    torch.manual_seed(31)
    num_heads, head_dim, width = 4, 64, 4
    hidden_dim = num_heads * head_dim
    batch_size, seq_len = 2, 128
    num_actual_tokens = batch_size * seq_len

    fused = (torch.randn(num_actual_tokens, 3 * hidden_dim, device=DEVICE) * 0.3).to(
        torch.bfloat16
    )
    strided = fused.split(hidden_dim, dim=-1)
    dense = tuple(x.contiguous() for x in strided)
    assert strided[0].stride(0) == 3 * hidden_dim

    raw_gate = (
        torch.randn(1, num_actual_tokens, num_heads, head_dim, device=DEVICE) * 0.3
    ).to(torch.bfloat16)
    raw_beta = torch.randn(1, num_actual_tokens, num_heads, device=DEVICE)
    weights = [
        torch.randn(hidden_dim, width, dtype=torch.float32, device=DEVICE) * 0.2
        for _ in range(3)
    ]
    a_log = torch.randn(num_heads, device=DEVICE) * 0.2
    dt_bias = torch.randn(hidden_dim, device=DEVICE) * 0.2
    query_start_loc = torch.arange(
        0, num_actual_tokens + 1, seq_len, device=DEVICE, dtype=torch.int32
    )
    state_indices = torch.arange(batch_size, device=DEVICE, dtype=torch.int32)
    has_initial_state = torch.zeros(batch_size, device=DEVICE, dtype=torch.bool)

    results = []
    for projections in (strided, dense):
        conv_state = torch.zeros(batch_size, 3 * hidden_dim, width - 1, device=DEVICE)
        recurrent_state = torch.zeros(
            batch_size, num_heads, head_dim, head_dim, device=DEVICE
        )
        output = torch.zeros(
            1,
            num_actual_tokens,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        torch.ops._xpu_C.kda_attention(
            output,
            *projections,
            raw_gate,
            raw_beta,
            conv_state,
            recurrent_state,
            *weights,
            a_log,
            dt_bias,
            batch_size,
            0,
            0,
            has_initial_state,
            query_start_loc,
            None,
            state_indices,
            None,
            None,
            None,
            None,
            num_actual_tokens,
            -5.0,
        )
        results.append((output, conv_state, recurrent_state))

    for from_strided, from_dense in zip(*results):
        torch.testing.assert_close(from_strided, from_dense, atol=0.0, rtol=0.0)


@requires_kda
@torch.inference_mode()
def test_kda_attention_honours_gate_lower_bound():
    """``gate_lower_bound`` selects the bounded sigmoid gate.

    Kimi-K3 configures ``-5.0``; passing ``None`` silently selects softplus, so
    the two must not agree.
    """
    torch.manual_seed(41)
    num_heads, head_dim, width = 4, 64, 4
    hidden_dim = num_heads * head_dim
    num_actual_tokens = 64

    common = dict(device=DEVICE)
    projections = tuple(
        (torch.randn(num_actual_tokens, hidden_dim, **common) * 0.3).to(torch.bfloat16)
        for _ in range(3)
    )
    raw_gate = (
        torch.randn(1, num_actual_tokens, num_heads, head_dim, **common) * 0.3
    ).to(torch.bfloat16)
    raw_beta = torch.randn(1, num_actual_tokens, num_heads, **common)
    weights = [
        torch.randn(hidden_dim, width, dtype=torch.float32, **common) * 0.2
        for _ in range(3)
    ]
    a_log = torch.randn(num_heads, **common) * 0.2
    dt_bias = torch.randn(hidden_dim, **common) * 0.2
    query_start_loc = torch.tensor(
        [0, num_actual_tokens], device=DEVICE, dtype=torch.int32
    )
    state_indices = torch.zeros(1, device=DEVICE, dtype=torch.int32)
    has_initial_state = torch.zeros(1, device=DEVICE, dtype=torch.bool)

    outputs = []
    for lower_bound in (None, -5.0):
        conv_state = torch.zeros(1, 3 * hidden_dim, width - 1, **common)
        recurrent_state = torch.zeros(1, num_heads, head_dim, head_dim, **common)
        output = torch.zeros(
            1,
            num_actual_tokens,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        torch.ops._xpu_C.kda_attention(
            output,
            *projections,
            raw_gate,
            raw_beta,
            conv_state,
            recurrent_state,
            *weights,
            a_log,
            dt_bias,
            1,
            0,
            0,
            has_initial_state,
            query_start_loc,
            None,
            state_indices,
            None,
            None,
            None,
            None,
            num_actual_tokens,
            lower_bound,
        )
        outputs.append(output.float())

    assert not torch.allclose(outputs[0], outputs[1], atol=1e-3)
