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
from abc import ABC, abstractmethod

import torch

from .types import (
    DecodeBatch,
    DecodeResult,
    ModelConfig,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)


class ModelRunner(ABC):
    """Runtime interface for compiled kernels registered to one model."""

    def __init__(self) -> None:
        self._kv_key_pages: dict[str, torch.Tensor] = {}
        self._kv_value_pages: dict[str, torch.Tensor] = {}
        self._kv_page_sizes: dict[str, int] = {}

    def init_kv_cache(
        self, model_id: str, config: ModelConfig, runtime: RuntimeConfig
    ) -> None:
        """Create the paged KV cache tensor pool for a model."""
        if model_id in self._kv_key_pages:
            return
        max_blocks_per_seq = math.ceil(runtime.max_seq_len / runtime.page_size)
        num_pages = runtime.total_kv_pages
        if num_pages is None:
            num_pages = runtime.max_batch_size * max_blocks_per_seq
        kv_dtype = getattr(torch, runtime.kv_dtype)
        key_pages = torch.zeros(
            config.num_hidden_layers,
            num_pages,
            config.num_key_value_heads,
            runtime.page_size,
            config.head_dim,
            dtype=kv_dtype,
            device=runtime.device,
        )
        value_pages = torch.zeros_like(key_pages)
        self._kv_key_pages[model_id] = key_pages
        self._kv_value_pages[model_id] = value_pages
        self._kv_page_sizes[model_id] = runtime.page_size

    def materialize_single_layer_cache(
        self, model_id: str, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-copy views for one layer's KV cache."""
        key_pages = self._kv_key_pages[model_id]
        value_pages = self._kv_value_pages[model_id]
        head_dim = key_pages.shape[-1]
        return (
            key_pages[layer_idx].reshape(-1, head_dim),
            value_pages[layer_idx].reshape(-1, head_dim),
        )

    def materialize_full_layer_cache(
        self, model_id: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-copy views spanning all layers' KV cache."""
        key_pages = self._kv_key_pages[model_id]
        value_pages = self._kv_value_pages[model_id]
        head_dim = key_pages.shape[-1]
        return (
            key_pages.reshape(-1, head_dim),
            value_pages.reshape(-1, head_dim),
        )

    @abstractmethod
    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run the compiled prefill path for one batch."""
        raise NotImplementedError

    @abstractmethod
    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run the compiled decode path for one batch."""
        raise NotImplementedError
