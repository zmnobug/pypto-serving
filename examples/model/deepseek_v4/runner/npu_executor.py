# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import operator
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

from examples.model.deepseek_v4.runner.npu_runner import (
    DEEPSEEK_V4_CSA_INNER_OUT_DIM,
    DEEPSEEK_V4_CSA_INNER_STATE_DIM,
    DEEPSEEK_V4_CSA_MAIN_OUT_DIM,
    DEEPSEEK_V4_CSA_STATE_DIM,
    DEEPSEEK_V4_HCA_MAIN_OUT_DIM,
    DEEPSEEK_V4_HCA_STATE_DIM,
    DEEPSEEK_V4_HC_MULT,
    DEEPSEEK_V4_IDX_HEAD_DIM,
    DeepSeekV4CacheLayout,
    DeepSeekV4CompiledKernels,
    DeepSeekV4L3Callable,
    DeepSeekV4ModelRunner,
    _DECODE_FWD_TENSOR_ORDER,
    _PREFILL_FWD_TENSOR_ORDER,
    build_deepseek_v4_layer_plan,
    DEEPSEEK_V4_CSA_NUM_LAYERS,
    DEEPSEEK_V4_FWD_NUM_LAYERS,
    DEEPSEEK_V4_HCA_NUM_LAYERS,
)
from examples.model.deepseek_v4.runner.weight_loader import DeepSeekV4WeightStore
from python.core.model_runner import ModelRunner
from python.core.pypto_executor import PyptoExecutor as CorePyptoExecutor
from python.core.types import RuntimeModel


_AST_INT_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
}
# CSA-group (x21) and HCA-group (x20) layer-stacked weight names emitted by the
# per-layer common dummy builder. Everything else there is a FWD weight (x43).
# Shared single-copy inputs (freqs/input_ids) are handled explicitly, not stacked.
_DECODE_FWD_CSA_STACKED_NAMES = frozenset(
    {
        "csa_cmp_wkv",
        "csa_cmp_wgate",
        "csa_cmp_ape",
        "csa_cmp_norm_w",
        "csa_idx_wq_b",
        "csa_idx_wq_b_scale",
        "csa_weights_proj",
        "csa_hadamard_idx",
        "csa_inner_wkv",
        "csa_inner_wgate",
        "csa_inner_ape",
        "csa_inner_norm_w",
    }
)
_DECODE_FWD_HCA_STACKED_NAMES = frozenset(
    {
        "hca_cmp_wkv",
        "hca_cmp_wgate",
        "hca_cmp_ape",
        "hca_cmp_norm_w",
    }
)
_DECODE_FWD_SHARED_COMMON_NAMES = frozenset({"freqs_cos", "freqs_sin", "input_ids"})
# Packed prefill now mirrors decode: the RoPE tables and input ids are passed as a
# single per-rank copy (the kernel slices them per layer internally), not stacked
# across the 43 forward layers.
_PREFILL_FWD_SHARED_COMMON_NAMES = frozenset({"freqs_cos", "freqs_sin", "input_ids"})
_DEEPSEEK_V4_IMPORT_MODULES = (
    "config",
    "moe",
    "combine",
    "decode_attention_csa",
    "decode_attention_hca",
    "decode_attention_swa",
    "decode_fwd",
    "decode_indexer",
    "decode_indexer_compressor",
    "decode_layer",
    "decode_sparse_attn",
    "decode_sparse_attn_csa",
    "decode_sparse_attn_hca",
    "decode_sparse_attn_swa",
    "dispatch",
    "expert_routed",
    "expert_shared",
    "gate",
    "hc_post",
    "hc_pre",
    "prefill_attention_csa",
    "prefill_attention_hca",
    "prefill_attention_swa",
    "prefill_indexer_compressor",
    "prefill_layer",
    "prefill_fwd",
    "prefill_sparse_attn",
    "qkv_proj_rope",
    "rmsnorm",
    "rope_tables",
)


def _find_pypto_lib_deepseek_v4_dir(pypto_root: str | None = None) -> Path:
    """Find the DeepSeekV4 kernel directory."""
    if pypto_root is None:
        pypto_root = os.environ.get("PYPTO_ROOT")
    if pypto_root:
        root = Path(pypto_root)
        candidate = root / "models" / "deepseek" / "v4"
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"DeepSeekV4 kernel directory not found under PYPTO_ROOT={pypto_root!r}")

    start_dir = Path(__file__).resolve().parent
    for directory in (start_dir, *start_dir.parents):
        pypto_lib_dir = directory / "pypto-lib"
        candidate = pypto_lib_dir / "models" / "deepseek" / "v4"
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Cannot locate DeepSeekV4 kernels. Run from a checkout with pypto-lib available "
        "or set PYPTO_ROOT to a pypto-lib checkout."
    )


def _int_constant_from_file(path: Path, name: str) -> int | None:
    """Read a simple integer module constant without importing kernel code."""
    tree = ast.parse(path.read_text(), filename=str(path))
    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    config_assignments = None

    def _eval_int(node: ast.AST) -> int | None:
        nonlocal config_assignments
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.Name):
            if node.id in assignments:
                return _eval_int(assignments[node.id])
            if config_assignments is None:
                config_path = path.parent / "config.py"
                if config_path == path or not config_path.exists():
                    config_assignments = {}
                else:
                    config_tree = ast.parse(config_path.read_text(), filename=str(config_path))
                    config_assignments = {
                        target.id: cfg_node.value
                        for cfg_node in config_tree.body
                        if isinstance(cfg_node, ast.Assign)
                        for target in cfg_node.targets
                        if isinstance(target, ast.Name)
                    }
            config_node = config_assignments.get(node.id)
            return _eval_int(config_node) if config_node is not None else None
        if isinstance(node, ast.BinOp):
            left = _eval_int(node.left)
            right = _eval_int(node.right)
            op = _AST_INT_OPERATORS.get(type(node.op))
            if left is None or right is None or op is None:
                return None
            return int(op(left, right))
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        return _eval_int(node.value)
    return None


def _is_deepseek_v4_module_file(path: Path, kernel_dir: Path) -> bool:
    """Return whether ``path`` is one of the top-level DeepSeekV4 kernel modules."""
    resolved = path.resolve()
    if resolved.is_relative_to(kernel_dir):
        return True
    parts = resolved.parts
    return len(parts) >= 4 and parts[-4:-1] == ("models", "deepseek", "v4")


@contextlib.contextmanager
def _deepseek_v4_import_context(
    kernel_dir: Path,
    *,
    pypto_root: Path,
    ep: int,
    moe_shape: str | None = None,
    num_layers: int | None = None,
):
    """Temporarily import DeepSeekV4 pypto-lib modules with a fixed EP argv."""
    old_argv = list(sys.argv)
    old_path = list(sys.path)
    missing = object()
    old_modules = {
        module_name: sys.modules.get(module_name, missing)
        for module_name in _DEEPSEEK_V4_IMPORT_MODULES
    }
    for module_name in _DEEPSEEK_V4_IMPORT_MODULES:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is not None and _is_deepseek_v4_module_file(Path(module_file), kernel_dir):
            sys.modules.pop(module_name, None)
    sys.argv = ["pypto-serving-deepseek-v4", "--ep", str(int(ep))]
    if moe_shape is not None:
        sys.argv.extend(["--moe-shape", moe_shape])
    if num_layers is not None:
        # prefill_fwd freezes its layer-stack span from ``--num-layers`` at import;
        # serving always packs the full 43-layer forward.
        sys.argv.extend(["--num-layers", str(int(num_layers))])
    sys.path.insert(0, str(kernel_dir))
    sys.path.insert(0, str(pypto_root))
    try:
        yield
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
        for module_name, module in old_modules.items():
            if module is missing:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = module


class DeepSeekV4PyptoExecutor(CorePyptoExecutor):
    """PyPTO executor boundary for DeepSeekV4 Flash W8A8 serving."""

    def __init__(
        self,
        kv_cache_manager=None,
        *,
        platform: str = "a2a3sim",
        device_id: int = 0,
        device_ids: Sequence[int] | None = None,
        save_kernels_dir: str | None = None,
        pypto_root: str | None = None,
        compile_kernels: bool = False,
        l3_trace: bool = False,
    ) -> None:
        worker_device_ids = tuple(device_ids) if device_ids is not None else (int(device_id),)
        super().__init__(
            kv_cache_manager,
            platform=platform,
            device_ids=worker_device_ids,
            save_kernels_dir=save_kernels_dir,
        )
        self._pypto_root = pypto_root
        self._kernel_dir = _find_pypto_lib_deepseek_v4_dir(pypto_root)
        self._compile_kernels = bool(compile_kernels)
        self._l3_trace = l3_trace
        self._embedding_cache: dict[str, torch.Tensor] = {}

    @property
    def profile_verbose(self) -> bool:
        """Return whether compile and L3 execution timing logs are enabled."""
        return self._l3_trace

    def lookup_embeddings(self, model: RuntimeModel, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup token embeddings from the lazily loaded DeepSeekV4 embedding table."""
        compiled = self._compiled.get(model.config.model_id)
        if not isinstance(compiled, DeepSeekV4CompiledKernels):
            raise RuntimeError(f"DeepSeekV4 model {model.config.model_id!r} is not registered")
        embed_weight = self._embedding_cache.get(model.config.model_id)
        if embed_weight is None:
            embed_weight = compiled.weight_store.load_tensor("embed.weight").contiguous()
            if embed_weight.ndim != 2:
                raise ValueError(f"embed.weight must be rank-2, got shape={tuple(embed_weight.shape)}")
            if int(embed_weight.shape[0]) != model.config.vocab_size:
                raise ValueError(
                    f"embed.weight vocab size must be {model.config.vocab_size}, "
                    f"got {int(embed_weight.shape[0])}"
                )
            if int(embed_weight.shape[1]) != model.config.hidden_size:
                raise ValueError(
                    f"embed.weight hidden size must be {model.config.hidden_size}, "
                    f"got {int(embed_weight.shape[1])}"
                )
            self._embedding_cache[model.config.model_id] = embed_weight

        flat_ids = token_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        embeddings = embed_weight.index_select(0, flat_ids)
        return embeddings.reshape(*token_ids.shape, model.config.hidden_size).to(device=token_ids.device)

    def release_finished_requests(self, request_ids: Iterable[str]) -> None:
        """Release runner-owned DeepSeekV4 cache slots for finished requests."""
        for runner in self._runners.values():
            release = getattr(runner, "release_finished_requests", None)
            if callable(release):
                release(request_ids)

    def _create_runner(self, model_id: str, compiled: object) -> ModelRunner:
        """Create the DeepSeekV4 runtime runner."""
        if not isinstance(compiled, DeepSeekV4CompiledKernels):
            raise TypeError("DeepSeekV4PyptoExecutor requires DeepSeekV4 compiled metadata.")
        return DeepSeekV4ModelRunner(compiled=compiled)

    def _compile_model(self, model: RuntimeModel) -> DeepSeekV4CompiledKernels:
        """Validate DeepSeekV4 W8A8 metadata and return runner artifacts.

        The current pypto-lib DeepSeekV4 programs are single-layer kernels. This
        method intentionally validates and packages the serving contract without
        pretending those kernels are already a full-model generator.
        """
        metadata = model.extra
        if metadata.get("family") != "deepseek_v4":
            raise ValueError("DeepSeekV4PyptoExecutor received a non-DeepSeekV4 model")
        if metadata.get("checkpoint_format") != "w8a8-compressed-tensors":
            raise ValueError("DeepSeekV4PyptoExecutor requires the W8A8 compressed-tensors checkpoint")

        layout = DeepSeekV4CacheLayout()
        layout.validate_runtime(model.config, model.runtime, self._device_ids)
        self._validate_kernel_contract(layout)
        compress_ratios = tuple(int(ratio) for ratio in metadata["compress_ratios"])
        if len(compress_ratios) != model.config.num_hidden_layers + 1:
            raise ValueError("DeepSeekV4 compress_ratios must include hidden layers plus MTP/final entry")
        config_data = metadata.get("config_data", {})
        n_routed_experts = int(config_data.get("n_routed_experts", 256)) if isinstance(config_data, dict) else 256
        num_hash_layers = int(config_data.get("num_hash_layers", 3)) if isinstance(config_data, dict) else 3
        layer_plan = build_deepseek_v4_layer_plan(
            compress_ratios=compress_ratios,
            num_hidden_layers=model.config.num_hidden_layers,
            num_hash_layers=num_hash_layers,
        )
        weight_map = dict(metadata["weight_map"])
        weight_store = DeepSeekV4WeightStore(model_dir=str(metadata["model_dir"]), weight_map=weight_map)
        weight_store.validate_startup_contract(
            num_hidden_layers=model.config.num_hidden_layers,
            n_routed_experts=n_routed_experts,
            compress_ratios=compress_ratios,
            num_hash_layers=num_hash_layers,
        )

        prefill = None
        decode = None
        freqs_cos = freqs_sin = None
        if self._compile_kernels:
            modules = self._load_kernel_modules(layout)
            prefill = self._compile_l3_callable(
                "deepseek_v4_prefill",
                modules["prefill_fwd"].l3_prefill_fwd,
                self._prefill_dummy_args(model, layout, modules["config"]),
            )
            decode = self._compile_l3_callable(
                "deepseek_v4_decode",
                modules["decode_fwd"].l3_decode_fwd,
                self._decode_dummy_args(model, layout, modules["config"]),
            )
            freqs_cos, freqs_sin = self._build_rope_tables(modules["rope_tables"], modules["config"])

        return DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir=str(metadata["model_dir"]),
            weight_map=weight_map,
            weight_store=weight_store,
            compress_ratios=compress_ratios,
            layer_plan=layer_plan,
            kernel_dir=str(self._kernel_dir),
            prefill=prefill,
            decode=decode,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            platform=self._platform,
            device_id=self._device_ids[0],
            n_routed_experts=n_routed_experts,
            num_hash_layers=num_hash_layers,
        )

    def _load_kernel_modules(self, layout: DeepSeekV4CacheLayout) -> dict[str, object]:
        """Import DeepSeekV4 pypto-lib modules with EP fixed to the serving world size."""
        pypto_root = self._kernel_dir.parents[2]
        ranks = layout.ranks
        fwd_layers = DEEPSEEK_V4_FWD_NUM_LAYERS
        with _deepseek_v4_import_context(
            self._kernel_dir,
            pypto_root=pypto_root,
            ep=ranks,
            moe_shape="prefill",
            num_layers=fwd_layers,
        ):
            prefill_layer = importlib.import_module("prefill_layer")
            prefill_fwd = importlib.import_module("prefill_fwd")
        with _deepseek_v4_import_context(self._kernel_dir, pypto_root=pypto_root, ep=ranks, moe_shape="decode"):
            modules = {
                name: importlib.import_module(name)
                for name in ("config", "decode_layer", "decode_fwd", "rope_tables")
            }
        modules["prefill_layer"] = prefill_layer
        modules["prefill_fwd"] = prefill_fwd
        return modules

    def _compile_l3_callable(self, name: str, jit_fn: object, dummy_args: Sequence[Any]) -> DeepSeekV4L3Callable:
        """Compile one DeepSeekV4 HOST wrapper into a distributed program."""
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
        from pypto.runtime import RunConfig  # noqa: PLC0415

        config = self._run_config(codegen_only=True)
        distributed_config = DistributedConfig(
            device_ids=list(self._device_ids),
            num_sub_workers=0,
        )
        run_config = RunConfig(
            platform=config.platform,
            device_id=config.device_id,
            backend_type=config.backend_type,
            strategy=config.strategy,
            dump_passes=config.dump_passes,
            save_kernels=config.save_kernels,
            save_kernels_dir=config.save_kernels_dir,
            codegen_only=True,
            pto_isa_commit=config.pto_isa_commit,
            diagnostic_phase=config.diagnostic_phase,
            disabled_diagnostics=config.disabled_diagnostics,
            compile_profiling=config.compile_profiling,
            enable_scope_stats=True,
            distributed_config=distributed_config,
        )
        compiled = jit_fn.compile(*dummy_args, config=run_config)
        if not isinstance(compiled, DistributedCompiledProgram):
            raise TypeError(f"{name} did not compile to DistributedCompiledProgram; got {type(compiled).__name__}")
        return DeepSeekV4L3Callable(compiled=compiled, name=name)

