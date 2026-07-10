# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from pypto.runtime import DeviceTensor

from examples.model.deepseek_v4.runner.weight_loader import DeepSeekV4WeightStore
from examples.model.deepseek_v4.runner.weight_loader import DeepSeekV4GlobalWeights
from examples.model.deepseek_v4.runner.weight_loader import DeepSeekV4StackedLayerWeights
from python.core.model_runner import ModelRunner
from python.core.types import (
    DecodeBatch,
    DecodeResult,
    ModelConfig,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)


DEEPSEEK_V4_RANKS = 8
DEEPSEEK_V4_HC_MULT = 4
DEEPSEEK_V4_BLOCK_SIZE = 128
DEEPSEEK_V4_DECODE_BATCH = 8
DEEPSEEK_V4_DECODE_SEQ = 1
DEEPSEEK_V4_DECODE_TOKENS = DEEPSEEK_V4_DECODE_BATCH * DEEPSEEK_V4_DECODE_SEQ
DEEPSEEK_V4_PREFILL_BATCH = 1
DEEPSEEK_V4_PREFILL_SEQ = 128
DEEPSEEK_V4_ORI_MAX_BLOCKS = 1
DEEPSEEK_V4_CMP_MAX_BLOCKS = 32
DEEPSEEK_V4_IDX_MAX_BLOCKS = 64
DEEPSEEK_V4_HCA_STATE_MAX_BLOCKS = 64
DEEPSEEK_V4_CSA_STATE_MAX_BLOCKS = 65
DEEPSEEK_V4_CSA_INNER_STATE_MAX_BLOCKS = 65
DEEPSEEK_V4_C128_STATE_BLOCK_SIZE = 8
DEEPSEEK_V4_C4_STATE_BLOCK_SIZE = 4
DEEPSEEK_V4_PREFILL_CMP_MAX_BLOCKS = DEEPSEEK_V4_CMP_MAX_BLOCKS
DEEPSEEK_V4_PREFILL_IDX_MAX_BLOCKS = DEEPSEEK_V4_IDX_MAX_BLOCKS
DEEPSEEK_V4_PREFILL_HCA_STATE_MAX_BLOCKS = 2048
DEEPSEEK_V4_PREFILL_CSA_STATE_MAX_BLOCKS = 4096
DEEPSEEK_V4_PREFILL_CSA_INNER_STATE_MAX_BLOCKS = 4096
DEEPSEEK_V4_INDEX_TOPK = 512
DEEPSEEK_V4_PREFILL_SPARSE_TOPK = DEEPSEEK_V4_BLOCK_SIZE + DEEPSEEK_V4_INDEX_TOPK
DEEPSEEK_V4_HEAD_DIM = 512
DEEPSEEK_V4_IDX_HEAD_DIM = 128
DEEPSEEK_V4_HCA_MAIN_OUT_DIM = 512
DEEPSEEK_V4_CSA_MAIN_OUT_DIM = 1024
DEEPSEEK_V4_CSA_INNER_OUT_DIM = 256
DEEPSEEK_V4_HCA_STATE_DIM = 2 * DEEPSEEK_V4_HCA_MAIN_OUT_DIM
DEEPSEEK_V4_CSA_STATE_DIM = 2 * DEEPSEEK_V4_CSA_MAIN_OUT_DIM
DEEPSEEK_V4_CSA_INNER_STATE_DIM = 2 * DEEPSEEK_V4_CSA_INNER_OUT_DIM
DEEPSEEK_V4_RMS_NORM_EPS = 1e-6
DEEPSEEK_V4_HC_EPS = 1e-6
# Layer-stacking counts for the packed all-layer decode_fwd kernel.
DEEPSEEK_V4_FWD_NUM_LAYERS = 43
DEEPSEEK_V4_CSA_NUM_LAYERS = 21
DEEPSEEK_V4_HCA_NUM_LAYERS = 20


# Argument order for the packed all-43-layer ``l3_prefill_fwd`` kernel. This
# mirrors pypto-lib prefill_fwd.py ``l3_prefill_fwd`` host signature: every
# layer-stacked weight/state tensor in core-parameter order, followed by the
# ``hc_head`` collapse weights, final RMSNorm input and an ``x_out`` output. The
# kernel stops before LM-head; logits are computed on the host from selected
# normalized hidden rows. A trailing ``num_tokens`` scalar is appended at dispatch.
# The work caches
# (kv_cache/cmp_kv/idx_kv_cache) are kernel ``pl.Out`` tensors; weights and
# metadata are inputs.
_PREFILL_FWD_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "kv_cache",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "cmp_kv",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_cmp_kv_state",
    "hca_cmp_score_state",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_cmp_kv_state",
    "csa_cmp_score_state",
    "csa_hadamard_idx",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_kv_state",
    "csa_inner_score_state",
    "idx_kv_cache",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "freqs_cos",
    "freqs_sin",
    "ori_block_table",
    "cmp_block_table",
    "idx_block_table",
    "ori_slot_mapping",
    "position_ids",
    "input_ids",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "cmp_sparse_indices",
    "cmp_sparse_lens",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
    "final_norm_w",
    "x_out",
)

# Argument order for the packed all-43-layer ``l3_decode_fwd`` kernel. This
# mirrors pypto-lib decode_fwd.py ``l3_decode_fwd`` host signature: after the
# ``hc_head`` collapse weights the kernel performs final RMSNorm and writes
# normalized ``x_out``. LM-head is computed on the host side.
_DECODE_FWD_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "kv_cache",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_compress_state",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_hadamard_idx",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_compress_state",
    "cmp_kv",
    "idx_kv_cache",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "freqs_cos",
    "freqs_sin",
    "block_table",
    "ori_slot_mapping",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "position_ids",
    "kv_seq_lens",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "cmp_block_table",
    "idx_block_table",
    "input_ids",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
    "final_norm_w",
    "x_out",
)

_DECODE_INPUT_TENSOR_FIELDS = (
    "input_ids",
    "position_ids",
    "kv_seq_lens",
    "block_table",
    "ori_slot_mapping",
    "cmp_block_table",
    "idx_block_table",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
)


@dataclass(frozen=True)
class DeepSeekV4CacheLayout:
    """Static cache layout baked into the current DeepSeekV4 kernels."""

    ranks: int = DEEPSEEK_V4_RANKS
    hc_mult: int = DEEPSEEK_V4_HC_MULT
    block_size: int = DEEPSEEK_V4_BLOCK_SIZE
    decode_batch: int = DEEPSEEK_V4_DECODE_BATCH
    decode_seq: int = DEEPSEEK_V4_DECODE_SEQ
    decode_tokens: int = DEEPSEEK_V4_DECODE_TOKENS
    prefill_batch: int = DEEPSEEK_V4_PREFILL_BATCH
    prefill_seq: int = DEEPSEEK_V4_PREFILL_SEQ
    ori_max_blocks: int = DEEPSEEK_V4_ORI_MAX_BLOCKS
    cmp_max_blocks: int = DEEPSEEK_V4_CMP_MAX_BLOCKS
    idx_max_blocks: int = DEEPSEEK_V4_IDX_MAX_BLOCKS
    hca_state_max_blocks: int = DEEPSEEK_V4_HCA_STATE_MAX_BLOCKS
    csa_state_max_blocks: int = DEEPSEEK_V4_CSA_STATE_MAX_BLOCKS
    csa_inner_state_max_blocks: int = DEEPSEEK_V4_CSA_INNER_STATE_MAX_BLOCKS
    c128_state_block_size: int = DEEPSEEK_V4_C128_STATE_BLOCK_SIZE
    c4_state_block_size: int = DEEPSEEK_V4_C4_STATE_BLOCK_SIZE
    prefill_cmp_max_blocks: int = DEEPSEEK_V4_PREFILL_CMP_MAX_BLOCKS
    prefill_idx_max_blocks: int = DEEPSEEK_V4_PREFILL_IDX_MAX_BLOCKS
    prefill_hca_state_max_blocks: int = DEEPSEEK_V4_PREFILL_HCA_STATE_MAX_BLOCKS
    prefill_csa_state_max_blocks: int = DEEPSEEK_V4_PREFILL_CSA_STATE_MAX_BLOCKS
    prefill_csa_inner_state_max_blocks: int = DEEPSEEK_V4_PREFILL_CSA_INNER_STATE_MAX_BLOCKS
    prefill_sparse_topk: int = DEEPSEEK_V4_PREFILL_SPARSE_TOPK

    @property
    def prefill_cmp_block_num(self) -> int:
        """Physical cmp_kv blocks per layer in the packed prefill kernel."""
        return self.decode_batch * self.prefill_cmp_max_blocks

    @property
    def prefill_idx_block_num(self) -> int:
        """Physical idx_kv_cache blocks per CSA layer in the packed prefill kernel."""
        return self.decode_batch * self.prefill_idx_max_blocks

    def validate_runtime(self, config: ModelConfig, runtime: RuntimeConfig, device_ids: Sequence[int]) -> None:
        """Validate serving/runtime options against kernel-fixed dimensions."""
        if len(device_ids) != self.ranks:
            raise ValueError(f"DeepSeekV4 requires exactly {self.ranks} devices, got {len(device_ids)}")
        if runtime.page_size != self.block_size:
            raise ValueError(f"DeepSeekV4 kernels require page_size={self.block_size}, got {runtime.page_size}")
        if runtime.max_batch_size > self.decode_batch:
            raise ValueError(
                f"DeepSeekV4 decode kernels support at most {self.decode_batch} active rows, "
                f"got max_batch_size={runtime.max_batch_size}"
            )
        decode_state_capacity = self.csa_state_max_blocks * self.c4_state_block_size
        if runtime.max_seq_len > decode_state_capacity:
            raise ValueError(
                "DeepSeekV4 pypto-lib decode CSA state tables currently support at most "
                f"max_seq_len={decode_state_capacity}, got {runtime.max_seq_len}. "
                "Increase the decode CSA state table depth in pypto-lib before serving longer contexts."
            )
        if self.decode_tokens != self.decode_batch * self.decode_seq:
            raise ValueError("DeepSeekV4 layout decode_tokens must equal decode_batch * decode_seq")
        expected = {
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "vocab_size": 129280,
        }
        actual = {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "vocab_size": config.vocab_size,
        }
        if actual != expected:
            mismatch = ", ".join(f"{name}={actual[name]} expected {value}" for name, value in expected.items())
            raise ValueError("DeepSeekV4 W8A8 kernels require Flash shape: " + mismatch)


@dataclass
class DeepSeekV4CacheManager:
    """Request-to-cache-slot mapping and table builders for DeepSeekV4 kernels."""

    layout: DeepSeekV4CacheLayout = field(default_factory=DeepSeekV4CacheLayout)
    _request_to_slot: dict[str, int] = field(default_factory=dict)
    _free_slots: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._free_slots:
            self._free_slots = list(range(self.layout.decode_batch))

    @property
    def active_slots(self) -> dict[str, int]:
        """Return a copy of currently assigned request slots."""
        return dict(self._request_to_slot)

    @property
    def free_count(self) -> int:
        """Return the number of unassigned decode slots."""
        return len(self._free_slots)

    def allocate(self, request_id: str) -> int | None:
        """Assign a stable decode slot to ``request_id``."""
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        if not self._free_slots:
            return None
        slot = self._free_slots.pop(0)
        self._request_to_slot[request_id] = slot
        return slot

    def release(self, request_ids: Iterable[str]) -> None:
        """Release slots held by finished or aborted requests."""
        for request_id in request_ids:
            slot = self._request_to_slot.pop(request_id, None)
            if slot is not None and slot not in self._free_slots:
                self._free_slots.append(slot)
        self._free_slots.sort()

    def slots_for_request_ids(self, request_ids: Sequence[str]) -> list[int]:
        """Return assigned slots for request ids, allocating missing slots."""
        slots = []
        for request_id in request_ids:
            slot = self.allocate(request_id)
            if slot is None:
                raise RuntimeError("DeepSeekV4 cache slots exhausted")
            slots.append(slot)
        return slots

    def block_table(self, slots: Sequence[int], *, max_blocks: int) -> torch.Tensor:
        """Build a row-major block table for request-owned physical block ranges."""
        table = torch.empty((len(slots), max_blocks), dtype=torch.int32)
        for row, slot in enumerate(slots):
            start = int(slot) * max_blocks
            table[row].copy_(torch.arange(start, start + max_blocks, dtype=torch.int32))
        return table

    def slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        max_blocks: int,
        block_size: int | None = None,
        compress_ratio: int = 1,
    ) -> torch.Tensor:
        """Map logical token positions to physical cache rows for each request slot."""
        block_size = self.layout.block_size if block_size is None else int(block_size)
        if compress_ratio <= 0:
            raise ValueError("compress_ratio must be positive")
        capacity = max_blocks * block_size
        max_tokens = max((len(row) for row in positions), default=0)
        mapping = torch.full((len(slots), max_tokens), -1, dtype=torch.int64)
        for row, (slot, row_positions) in enumerate(zip(slots, positions, strict=True)):
            base = int(slot) * capacity
            for col, position in enumerate(row_positions):
                logical = int(position) // compress_ratio
                if logical >= capacity:
                    raise ValueError(
                        f"position {position} maps to logical cache row {logical}, "
                        f"but capacity is {capacity}"
                    )
                mapping[row, col] = base + logical
        return mapping

    def block_table_for_kernel_rows(
        self,
        slots: Sequence[int],
        *,
        max_blocks: int,
        kernel_rows: int,
    ) -> torch.Tensor:
        """Build a fixed-row block table, replicating row 0 into inactive rows."""
        if not slots:
            raise ValueError("slots must not be empty")
        active = self.block_table(slots, max_blocks=max_blocks)
        return self.replicate_first_row(active, actual_rows=len(slots), kernel_rows=kernel_rows)

    def sliding_window_slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        kernel_rows: int,
    ) -> torch.Tensor:
        """Map absolute positions into the 128-token ori sliding-window cache."""
        rows = self._replicated_slots_and_positions(slots, positions, kernel_rows=kernel_rows)
        mapping = torch.full((kernel_rows, max((len(row) for _, row in rows), default=0)), -1, dtype=torch.int64)
        for row_idx, (slot, row_positions) in enumerate(rows):
            base = int(slot) * self.layout.ori_max_blocks * self.layout.block_size
            for col, position in enumerate(row_positions):
                window_slot = int(position) % self.layout.block_size
                mapping[row_idx, col] = base + window_slot
        return mapping

    def compressed_slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        max_blocks: int,
        compress_ratio: int,
        kernel_rows: int,
    ) -> torch.Tensor:
        """Map compression-boundary positions into a compressed KV cache."""
        rows = self._replicated_slots_and_positions(slots, positions, kernel_rows=kernel_rows)
        mapping = torch.full((kernel_rows, max((len(row) for _, row in rows), default=0)), -1, dtype=torch.int64)
        capacity = max_blocks * self.layout.block_size
        for row_idx, (slot, row_positions) in enumerate(rows):
            base = int(slot) * capacity
            for col, position in enumerate(row_positions):
                position = int(position)
                if (position + 1) % compress_ratio != 0:
                    continue
                logical = position // compress_ratio
                if logical >= capacity:
                    raise ValueError(
                        f"position {position} maps to compressed row {logical}, "
                        f"but capacity is {capacity}"
                    )
                mapping[row_idx, col] = base + logical
        return mapping

    def state_slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        max_blocks: int,
        state_block_size: int,
        kernel_rows: int,
    ) -> torch.Tensor:
        """Map absolute token positions into a compressor-state cache."""
        rows = self._replicated_slots_and_positions(slots, positions, kernel_rows=kernel_rows)
        mapping = torch.full((kernel_rows, max((len(row) for _, row in rows), default=0)), -1, dtype=torch.int64)
        capacity = max_blocks * state_block_size
        for row_idx, (slot, row_positions) in enumerate(rows):
            base = int(slot) * capacity
            for col, position in enumerate(row_positions):
                position = int(position)
                if position >= capacity:
                    raise ValueError(
                        f"position {position} exceeds compressor-state capacity {capacity} "
                        f"(max_blocks={max_blocks}, state_block_size={state_block_size})"
                    )
                mapping[row_idx, col] = base + position
        return mapping

    @staticmethod
    def _replicated_slots_and_positions(
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        kernel_rows: int,
    ) -> list[tuple[int, Sequence[int]]]:
        if not slots:
            raise ValueError("slots must not be empty")
        if len(slots) != len(positions):
            raise ValueError("slots and positions must have the same active row count")
        if len(slots) > kernel_rows:
            raise ValueError("active rows exceed kernel_rows")
        rows = [(int(slot), tuple(int(pos) for pos in row)) for slot, row in zip(slots, positions, strict=True)]
        rows.extend((rows[0][0], rows[0][1]) for _ in range(kernel_rows - len(rows)))
        return rows

    @staticmethod
    def replicate_first_row(tensor: torch.Tensor, *, actual_rows: int, kernel_rows: int) -> torch.Tensor:
        """Pad kernel inputs by replicating row 0 into inactive rows."""
        if actual_rows <= 0:
            raise ValueError("actual_rows must be positive")
        if kernel_rows < actual_rows:
            raise ValueError("kernel_rows must be >= actual_rows")
        if tensor.shape[0] < actual_rows:
            raise ValueError("tensor has fewer rows than actual_rows")
        out = torch.empty((kernel_rows, *tensor.shape[1:]), dtype=tensor.dtype)
        out[:actual_rows].copy_(tensor[:actual_rows])
        if actual_rows < kernel_rows:
            out[actual_rows:].copy_(tensor[0:1].expand(kernel_rows - actual_rows, *tensor.shape[1:]))
        return out


