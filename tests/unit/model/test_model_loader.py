# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import builtins
import json
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.config.types import RuntimeConfig
from pypto_serving.model import model_loader


@pytest.fixture
def v41_imports(monkeypatch):
    """Track V4.1 imports even if another test has already cached the module."""
    original_import = builtins.__import__
    imports = []

    def checked_import(name, *args, **kwargs):
        if "deepseek_v41" in name:
            imports.append(name)
            raise ImportError("V4.1 backend is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.delattr(torch, "float8_e8m0fnu", raising=False)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    return imports


@pytest.mark.parametrize(
    ("family", "model_format", "loader_class"),
    [
        ("deepseek_v4", None, model_loader.DeepSeekV4W8A8DirectoryLoader),
        ("deepseek_v4", "dsv4-w8a8", model_loader.DeepSeekV4W8A8DirectoryLoader),
        ("qwen3", None, model_loader.HuggingFaceDirectoryLoader),
        ("qwen3", "hf", model_loader.HuggingFaceDirectoryLoader),
    ],
)
def test_other_models_do_not_import_v41(tmp_path, monkeypatch, v41_imports, family, model_format, loader_class):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": family}))
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    sentinel = object()
    monkeypatch.setattr(loader_class, "load", lambda self, request: sentinel)

    loaded = model_loader.ModelLoader().load("model", str(tmp_path), model_format=model_format)

    assert loaded is sentinel
    assert v41_imports == []
    assert not hasattr(torch, "float8_e8m0fnu")


@pytest.mark.parametrize("model_format", [None, "deepseek_v41", "deepseek-v41", "DSV41"])
def test_v41_imports_backend_only_when_loading(tmp_path, monkeypatch, model_format):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    original_import = builtins.__import__
    imports = []
    requests = []
    sentinel = object()

    class BackendLoader:
        def load(self, request):
            requests.append(request)
            return sentinel

    def backend_import(name, *args, **kwargs):
        if name.endswith("deepseek_v41.npu_executor"):
            imports.append(name)
            return SimpleNamespace(DeepSeekV41DirectoryLoader=BackendLoader)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", backend_import)
    loader = model_loader.ModelLoader()
    assert imports == []
    runtime = RuntimeConfig()

    loaded = loader.load("v41", str(tmp_path), runtime, model_format, trust_remote_code=True)

    assert loaded is sentinel
    assert len(imports) == 1
    assert requests == [model_loader.ModelLoadRequest(
        "v41", str(tmp_path), runtime, model_format, {"trust_remote_code": True},
    )]


@pytest.mark.parametrize("model_format", ["hf", "deepseek_v4_w8a8"])
def test_v41_rejects_other_formats_without_importing_backend(tmp_path, v41_imports, model_format):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))

    with pytest.raises(ValueError, match="a V4.1 checkpoint requires model_format='deepseek_v41'"):
        model_loader.ModelLoader().load("v41", str(tmp_path), model_format=model_format)

    assert v41_imports == []


def test_v41_backend_import_failure_is_not_hidden(tmp_path, v41_imports):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    (tmp_path / "model.safetensors.index.json").write_text("{}")

    with pytest.raises(ImportError, match="V4.1 backend is unavailable"):
        model_loader.ModelLoader().load("v41", str(tmp_path))

    assert len(v41_imports) == 1
