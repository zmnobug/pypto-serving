# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Opt-in observation of already-loaded modules in actual Serving workers.

No model/native modules are imported, patched or replaced by this observer.
Only the regression server sets DSPARK_REGRESSION_MANIFEST and this PYTHONPATH.
"""

import os
import sys


def _start():
    if not os.environ.get("DSPARK_REGRESSION_MANIFEST"):
        return
    import threading
    import time

    def observe():
        while True:
            try:
                eligible = False
                for frame in sys._current_frames().values():
                    while frame is not None:
                        if (frame.f_code.co_name == "_worker_entry"
                                and frame.f_globals.get("__name__", "").endswith(".serving_worker")
                                and "worker" in frame.f_locals):
                            eligible = True
                        frame = frame.f_back
                if eligible and all(name in sys.modules for name in ("pypto.pypto_core", "_task_interface")):
                    import json
                    from pathlib import Path
                    # Add only the harness package; never an alternate model source.
                    harness = str(Path(__file__).resolve().parents[2])
                    sys.path.append(harness)
                    from deepseek_agentx.identity import capture
                    path = Path(os.environ["DSPARK_REGRESSION_MANIFEST"])
                    manifest = json.loads(path.read_text())
                    manifest["output"] = str(path.parent / "environment/workers" / f"worker-{os.getpid()}.json")
                    capture(manifest, "actual-model-worker", with_imports=True, loaded_only=True)
                    return
            except Exception as exc:
                from pathlib import Path
                error = Path(os.environ["DSPARK_REGRESSION_MANIFEST"]).parent / "environment" / f"probe-{os.getpid()}.error"
                error.write_text(f"{type(exc).__name__}: {exc}\n")
            time.sleep(1)

    threading.Thread(target=observe, name="regression-identity", daemon=True).start()


if os.environ.get("DSPARK_REGRESSION_MANIFEST"):
    _start()
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_start)
