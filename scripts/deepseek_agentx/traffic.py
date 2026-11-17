# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local streaming HTTP observer for client/server request reconciliation."""

from contextlib import contextmanager
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import uuid

from .common import write_json

HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
               "trailers", "transfer-encoding", "upgrade", "content-length"}


@contextmanager
def observe_traffic(root, upstream_port, timeout):
    root = Path(root)
    (root / "replay/http").mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_args):
            pass

        def do_GET(self):
            # AgentX may probe /v1/models before sending measured requests.
            connection = http.client.HTTPConnection("127.0.0.1", upstream_port, timeout=min(timeout, 30))
            try:
                connection.request("GET", self.path)
                response = connection.getresponse()
                body = response.read(16 * 1024 * 1024 + 1)
                if len(body) > 16 * 1024 * 1024:
                    raise ValueError("Oversized readiness response")
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in HOP_HEADERS:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError):
                self.close_connection = True
            finally:
                connection.close()

        def do_POST(self):
            record = {"schema_version": 1, "client_request_id": self.headers.get("X-Request-ID"),
                      "path": self.path, "started_ns": time.time_ns(), "done": False,
                      "server_request_id": None, "finish_reason": None, "usage": None}
            connection = None
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0 or length > 64 * 1024 * 1024:
                    raise ValueError("Expected a bounded Content-Length")
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Incomplete request body")
                name = uuid.uuid4().hex + ".json"
                (root / "replay/http" / name).write_bytes(body)
                record["body_file"] = "replay/http/" + name
                record["body_sha256"] = hashlib.sha256(body).hexdigest()
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS | {"host"}}
                connection = http.client.HTTPConnection("127.0.0.1", upstream_port, timeout=timeout)
                connection.request("POST", self.path, body, headers)
                response = connection.getresponse()
                record["status"] = response.status
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in HOP_HEADERS:
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()
                buffer = b""
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == b"[DONE]":
                            record["done"] = True
                            continue
                        value = json.loads(payload)
                        if value.get("id"):
                            previous = record["server_request_id"]
                            if previous and previous != value["id"]:
                                raise ValueError("SSE response changed request id")
                            record["server_request_id"] = value["id"]
                        for choice in value.get("choices", []):
                            if choice.get("finish_reason"):
                                record["finish_reason"] = choice["finish_reason"]
                        if value.get("usage"):
                            record["usage"] = value["usage"]
                        if value.get("error"):
                            record["error"] = value["error"]
                    if len(buffer) > 16 * 1024 * 1024:
                        raise ValueError("Oversized SSE event")
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                # An upstream failure must remain visible to the client.
                self.close_connection = True
            finally:
                if connection:
                    connection.close()
                record["ended_ns"] = time.time_ns()
                with lock, (root / "replay/requests.jsonl").open("a") as output:
                    output.write(json.dumps(record, sort_keys=True) + "\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    write_json(root / "replay/observer.json", {"port": server.server_port, "upstream_port": upstream_port,
                                               "purpose": "streaming request ID/usage observation"})
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
