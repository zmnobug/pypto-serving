# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations


import pytest

import pypto_serving.cli.main as cli


def _parse_cli_args(argv: list[str]):
    return cli.build_parser().parse_args(argv)


@pytest.mark.parametrize(("role", "specified", "expected"), [
    (None, None, 16), ("prefill", None, 16), ("decode", None, 72),
    (None, 32, 32), ("prefill", 8, 8), ("decode", 16, 16),
])
def test_request_capacity_defaults_reach_scheduler_and_cache(tmp_path, role, specified, expected):
    model = tmp_path / "model"
    model.mkdir()
    argv = ["--model", str(model)]
    if role is not None:
        argv += ["--pd-role", role]
    if specified is not None:
        argv += ["--max-num-seqs", str(specified)]
    config = cli.build_serving_engine_config(_parse_cli_args(argv))
    assert config.max_num_running_reqs == expected
    assert config.runtime_config.max_batch_size == expected


def test_cli_keeps_generic_model_chunk_sizes_unrestricted():
    args = _parse_cli_args(
        ["--model", "model", "--long-prefill-token-threshold", "128"]
    )

    cli._validate_prefill_chunk_size("qwen", args.long_prefill_token_threshold)
    assert args.long_prefill_token_threshold == 128


def test_agentinfer_config_is_explicit_and_reaches_engine(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    options = '{"resume_capacity_ratio":0.85,"resume_wait_relax_seconds":8,' \
              '"resume_wait_growth_reserve_fraction":0.06,"resume_wait_extra_headroom_tokens":512,' \
              '"force_resume_timeout_min_seconds":1,' \
              '"force_resume_timeout_max_seconds":3}'
    args = _parse_cli_args([
        "--model", str(model_dir), "--agentinfer-mode", "enforce", "--agentinfer-config", options,
    ])
    config = cli.build_serving_engine_config(args)
    assert config.agentinfer_config.resume_capacity_ratio == 0.85
    assert config.agentinfer_config.resume_wait_relax_seconds == 8
    assert config.agentinfer_config.resume_wait_growth_reserve_fraction == 0.06
    assert config.agentinfer_config.resume_wait_extra_headroom_tokens == 512
    assert config.agentinfer_config.force_resume_timeout_max_seconds == 3
    args.agentinfer_mode = "disabled"
    with pytest.raises(ValueError, match="requires --agentinfer-mode enforce"):
        cli.build_serving_engine_config(args)
    args.agentinfer_config = []
    with pytest.raises(ValueError, match="JSON object"):
        cli.build_serving_engine_config(args)


def test_cli_keeps_dspark_chunk_sizes_under_its_kernel_contract():
    args = _parse_cli_args(
        ["--model", "model", "--long-prefill-token-threshold", "128"]
    )

    cli._validate_prefill_chunk_size(
        "deepseek_v4",
        args.long_prefill_token_threshold,
        variant="dspark",
    )
    assert args.long_prefill_token_threshold == 128


def test_build_serving_engine_config_rejects_unsupported_deepseek_chunk_size(
    tmp_path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"deepseek_v4"}')
    args = _parse_cli_args(
        ["--model", str(model_dir), "--long-prefill-token-threshold", "3072"]
    )

    with pytest.raises(
        ValueError,
        match="--long-prefill-token-threshold must be one of",
    ):
        cli.build_serving_engine_config(args)


def test_build_serving_engine_config_uses_parallel_config_for_devices(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = _parse_cli_args(
        [
            "--model",
            str(model_dir),
            "--devices",
            "0,1,2,3",
            "--dp",
            "2",
            "--tp",
            "2",
        ]
    )

    config = cli.build_serving_engine_config(args)

    assert config.device_id == 0
    assert config.device_ids == ()
    assert config.worker_device_ids() == (0, 1)
    assert config.parallel_config.data_parallel_size == 2
    assert config.parallel_config.tensor_parallel_size == 2
    assert config.parallel_config.replica_device_groups == ((0, 1), (2, 3))


def test_agentinfer_mode_is_explicit_and_single_replica_only(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert cli.build_serving_engine_config(
        _parse_cli_args(["--model", str(model_dir)])
    ).agentinfer_mode == "disabled"
    assert cli.build_serving_engine_config(
        _parse_cli_args(["--model", str(model_dir), "--agentinfer-mode", "enforce"])
    ).agentinfer_mode == "enforce"
    with pytest.raises(ValueError, match="single-replica Qwen"):
        cli.build_serving_engine_config(_parse_cli_args([
            "--model", str(model_dir), "--agentinfer-mode", "enforce",
            "--devices", "0,1", "--dp", "2",
        ]))


def test_agentinfer_cli_rejects_deepseek_v41(tmp_path):
    model_dir = tmp_path / "deepseek-v41"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"deepseek_v41"}')
    args = _parse_cli_args([
        "--model", str(model_dir), "--agentinfer-mode", "enforce",
    ])

    with pytest.raises(ValueError, match="DeepSeek V4.1 does not support AgentInfer"):
        cli.build_serving_engine_config(args)


def test_agentinfer_cli_accepts_only_standard_deepseek_k1(tmp_path, monkeypatch):
    from pypto_serving.config.types import RuntimeConfig

    model_dir = tmp_path / "deepseek"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"deepseek_v4"}')
    monkeypatch.setattr(cli, "_validate_model_topology", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_build_runtime_config", lambda *args, **kwargs: RuntimeConfig())
    common = [
        "--model", str(model_dir), "--devices", "0,1,2,3,4,5,6,7",
        "--dp", "8", "--ep", "8", "--agentinfer-mode", "enforce",
    ]
    k1 = common + ["--speculative-config", '{"method":"mtp","num_speculative_tokens":1}']
    config = cli.build_serving_engine_config(_parse_cli_args(k1))
    assert config.executor_cls == "PyptoDeepSeekV4Executor"
    assert config.agentinfer_mode == "enforce"

    for flags in (
        ["--speculative-config", '{"method":"mtp","num_speculative_tokens":3}'],
        ["--speculative-config", '{"method":"mtp","num_speculative_tokens":1}',
         "--no-enable-prefix-caching"],
    ):
        with pytest.raises(ValueError, match="MTP K=1 and prefix caching"):
            cli.build_serving_engine_config(_parse_cli_args(common + flags))


@pytest.mark.parametrize("num_speculative_tokens", [0, 7])
def test_agentinfer_cli_accepts_dspark_k0_and_k7(tmp_path, monkeypatch, num_speculative_tokens):
    from pypto_serving.config.types import RuntimeConfig

    model_dir = tmp_path / "dspark"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"deepseek_v4"}')
    monkeypatch.setattr(cli, "_validate_model_topology", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_build_runtime_config", lambda *args, **kwargs: RuntimeConfig())
    monkeypatch.setattr(cli, "_resolve_num_speculative_tokens", lambda _args: num_speculative_tokens)
    args = _parse_cli_args([
        "--model", str(model_dir),
        "--devices", ",".join(str(device) for device in range(16)),
        "--dp", "4", "--ep", "16", "--tp", "4",
        "--speculative-config",
        f'{{"method":"dspark","num_speculative_tokens":{num_speculative_tokens}}}',
        "--agentinfer-mode", "enforce",
    ])

    config = cli.build_serving_engine_config(args)

    assert config.executor_cls == "PyptoDeepSeekV4DSparkExecutor"
    assert config.agentinfer_mode == "enforce"


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("mode", ["disabled", "enforce"])
def test_pd_dspark_requires_agentinfer_disabled(tmp_path, monkeypatch, role, mode):
    from pypto_serving.config.types import RuntimeConfig

    model_dir = tmp_path / "dspark"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"deepseek_v4"}')
    monkeypatch.setattr(cli, "_validate_model_topology", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli, "_build_runtime_config", lambda *args, **kwargs: RuntimeConfig(num_speculative_tokens=7),
    )
    args = _parse_cli_args([
        "--model", str(model_dir),
        "--devices", ",".join(str(device) for device in range(16)),
        "--dp", "4", "--ep", "16", "--tp", "4", "--max-num-seqs", "4",
        "--speculative-config", '{"method":"dspark","num_speculative_tokens":7}',
        "--pd-role", role, "--pd-config", str(tmp_path / "pd.json"),
        "--agentinfer-mode", mode,
    ])

    if mode == "enforce":
        with pytest.raises(ValueError, match="PD deployment does not support --agentinfer-mode enforce"):
            cli.build_serving_engine_config(args)
    else:
        config = cli.build_serving_engine_config(args)
        assert config.executor_cls == "PyptoDeepSeekV4DSparkExecutor"
        assert config.agentinfer_mode == "disabled"


def test_agentinfer_cli_rejects_dspark_without_prefix_cache(tmp_path, monkeypatch):
    from pypto_serving.config.types import RuntimeConfig

    model_dir = tmp_path / "dspark"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"deepseek_v4"}')
    monkeypatch.setattr(cli, "_validate_model_topology", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_build_runtime_config", lambda *args, **kwargs: RuntimeConfig())
    monkeypatch.setattr(cli, "_resolve_num_speculative_tokens", lambda _args: 7)
    args = _parse_cli_args([
        "--model", str(model_dir),
        "--devices", ",".join(str(device) for device in range(16)),
        "--dp", "4", "--ep", "16", "--tp", "4",
        "--speculative-config", '{"method":"dspark","num_speculative_tokens":7}',
        "--no-enable-prefix-caching", "--agentinfer-mode", "enforce",
    ])

    with pytest.raises(ValueError, match="prefix caching enabled"):
        cli.build_serving_engine_config(args)


def test_build_serving_engine_config_rejects_invalid_parallel_topology(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = _parse_cli_args(
        [
            "--model",
            str(model_dir),
            "--devices",
            "0,1,2",
            "--dp",
            "2",
            "--tp",
            "2",
        ]
    )

    with pytest.raises(ValueError, match="number of devices"):
        cli.build_serving_engine_config(args)


def test_generate_config_rejects_wrong_value_types():
    for invalid in (
        {"stream": "false"},
        {"ignore_eos": "true"},
        {"stop": "END"},
        {"stop": [1, 2]},
        {"max_new_tokens": 2.5},
        {"max_new_tokens": True},
        {"max_new_tokens": 0},
        {"temperature": -0.1},
        {"temperature": "0"},
        {"top_p": 0},
        {"top_p": 1.5},
        {"top_k": 0},
        {"top_k": "5"},
    ):
        with pytest.raises(ValueError, match="--generate-config"):
            cli._build_generate_config(invalid)


def test_generate_config_accepts_valid_values():
    config = cli._build_generate_config(
        {
            "max_new_tokens": 8,
            "temperature": 0.2,
            "top_p": 1,
            "top_k": None,
            "stop": ["END"],
            "stream": True,
            "ignore_eos": False,
        }
    )

    assert config.max_new_tokens == 8
    assert config.temperature == 0.2
    assert config.top_p == 1
    assert config.top_k is None
    assert config.stop == ("END",)
    assert config.stream is True
    assert config.ignore_eos is False


@pytest.mark.parametrize("options", [None, {"max_new_tokens": 8}])
def test_generate_config_defaults_to_greedy(options):
    config = cli._build_generate_config(options)

    assert config.temperature == 0.0


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_dspark_topology_accepts_the_8_card_and_16_card_worlds(tp):
    # Attention TP determines cache partitions; expert parallelism spans the world.
    for ep in (8, 16):
        dp = ep // tp
        devices = ",".join(str(device) for device in range(ep))
        args = _parse_cli_args(
            [
                "--model", "model",
                "--dp", str(dp),
                "--ep", str(ep),
                "--tp", str(tp),
                "--devices", devices,
                "--block-size", "32",
                "--max-num-seqs", "8",
                "--max-model-len", "1024",
            ]
        )

        cli._validate_dspark_topology(args)


@pytest.mark.parametrize("requested", [None, "", "diagnostic-output"])
def test_worker_only_exports_explicit_build_directory(monkeypatch, requested):
    import os
    from pathlib import Path
    from types import SimpleNamespace

    from pypto_serving.serving.server.serving_worker import WorkerProcess

    if requested is None:
        monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
    else:
        monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", requested)
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.config = SimpleNamespace(dp_rank=2)
    directory = worker._configure_pypto_build_dir((3, 4))
    assert directory == (Path(requested) / "serving_dp2_d3_4" if requested else None)
    assert os.environ.get("PYPTO_PROG_BUILD_DIR") == (str(directory) if requested else requested)
