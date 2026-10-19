"""Minimal OpenAI-compatible HTTP server exposing the System One contract.

Endpoints:
  POST /v1/systemone         -> structured decisions (the vLLM v0.32 feature)
  POST /v1/chat/completions  -> stub chat endpoint: the "old way", where the
                                decision is buried in prose, for the
                                comparison demo in agent_demo.py
  GET  /v1/models            -> pseudo model card
  GET  /healthz              -> {"status": "ok"}

Standard library only. Run:  python3 server.py [port]   (default 8765)
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import systemone

MODEL_ID = "proxy-systemone-1"


def _chat_reply(prompt: str) -> str:
    """Stub chat behaviour with deliberately varied prose formatting.

    Two tickets get JSON embedded in prose (salvageable with a regex); one
    gets a free-form sentence that breaks naive parsers -- the failure mode
    the System One endpoint exists to eliminate.
    """
    low = prompt.lower()
    if "refund" in low or "charged twice" in low:
        return (
            "I have classified this ticket. Here is my answer:\n"
            '{"department": "billing", "urgency": "medium"}\n'
            "Let me know if you need anything else."
        )
    if "password" in low or "reset email" in low:
        return (
            "This looks like an account access issue to me, so I would route "
            "it to the account team with low urgency."
        )
    return '```json\n{"department": "engineering", "urgency": "high"}\n```'


class Handler(BaseHTTPRequestHandler):
    server_version = "SystemOneProxy/1.0"

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode() or "{}")
        except (ValueError, UnicodeDecodeError):
            return None

    def do_GET(self):  # noqa: N802
        if self.path == "/healthz":
            self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "system_one": True}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        body = self._read_json()
        if body is None:
            self._send(400, {"error": "invalid JSON body"})
            return
        if self.path == "/v1/systemone":
            try:
                status, resp = systemone.decide(body)
            except systemone.DecisionError as exc:
                self._send(exc.status, {"error": str(exc)})
                return
            self._send(status, resp)
        elif self.path == "/v1/chat/completions":
            messages = body.get("messages", []) if isinstance(body, dict) else []
            prompt = ""
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    prompt = str(msg.get("content", ""))
                    break
            reply = _chat_reply(prompt)
            self._send(200, {
                "id": "chatcmpl-proxy",
                "object": "chat.completion",
                "model": body.get("model", MODEL_ID) if isinstance(body, dict) else MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt.split()),
                          "completion_tokens": len(reply.split()),
                          "total_tokens": len(prompt.split()) + len(reply.split())},
            })
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):  # keep demo output clean
        pass


def create_server(port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    httpd = create_server(port)
    print(f"serving on http://127.0.0.1:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
