"""The student behind the ``/v1/systemone`` wire contract, so Factor adopts
it on its decision port and Smrti asks it over ``SMRTI_DECISIONS_URL``, the
same as the EdgeJev server it replaces.

Standard library only: on the machine this is for, a web framework's import
is a measurable share of the memory the model needs.
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..provider import DecisionUnavailable, DecisionUnsupported
from .runtime import HEAD_MAX_LEN, MAX_LEN, Student

logger = logging.getLogger("smrti.decisions.student.serve")

REQUEST_PATH = "/v1/systemone"
# The error type a refusal carries, so a client tells "not one of my
# questions" from a bad request.
UNSUPPORTED = "unsupported_question"
# A request is a state under the window and a few questions; anything past
# this is not one, and is refused before it is read into memory.
MAX_BODY = 1 << 20
# How long a caller may hold a handler thread without sending, and how long
# a request waits for the one graph before it is told to try later: a
# caller that has already given up on its deadline must not be served.
CLIENT_TIMEOUT = 30.0
QUEUE_TIMEOUT = 20.0


def _handler(student: Student, lock: threading.Lock) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = CLIENT_TIMEOUT

        def log_message(self, *_args: Any) -> None:
            pass

        def _send(self, code: int, body: Any) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                # The caller gave up on its deadline; nothing to tell it.
                pass

        def do_GET(self) -> None:
            if self.path.rstrip("/") in ("/health", "/ready"):
                # max_len and head_max_len are what a caller sizes its
                # state against: the encoder's window plus the head Laya
                # would carry and the student does not.
                self._send(200, {"ok": True, "model": student.model_name, "backend": "student",
                                 "precision": "int8", "max_len": MAX_LEN + HEAD_MAX_LEN,
                                 "head_max_len": HEAD_MAX_LEN})
                return
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != REQUEST_PATH:
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if not 0 <= n <= MAX_BODY:
                    raise ValueError(f"the request body must be 0 to {MAX_BODY} bytes")
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("the request body must be a JSON object")
                state, questions = body.get("state"), body.get("questions")
                if state is None or not isinstance(questions, dict) or not questions:
                    raise ValueError("a request needs a state and a questions object")
                if not lock.acquire(timeout=QUEUE_TIMEOUT):
                    self._send(503, {"error": {"message": "the student is busy", "type": "server_busy"}})
                    return
                try:
                    out = student.predict(state, questions)
                finally:
                    lock.release()
            except DecisionUnsupported as exc:
                self._send(400, {"error": {"message": str(exc)[:400], "type": UNSUPPORTED}})
                return
            except (ValueError, DecisionUnavailable) as exc:
                self._send(400, {"error": {"message": str(exc)[:400], "type": "invalid_request_error"}})
                return
            except Exception as exc:  # a bug, reported as such rather than as a refusal
                logger.exception("the student failed on a request")
                self._send(500, {"error": {"message": str(exc)[:400], "type": "server_error"}})
                return
            self._send(200, out)

    return Handler


def run_student_server(model: Path | str, host: str = "127.0.0.1", port: int = 8731,
                       threads: int | None = None) -> None:
    student = Student(model, threads=threads)
    # One graph, one caller at a time: the session holds the machine's
    # inference threads already, and two runs at once only slow both.
    server = ThreadingHTTPServer((host, port), _handler(student, threading.Lock()))
    logger.info("student decision server on http://%s:%d%s (%s)", host, port, REQUEST_PATH, student.model_name)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
