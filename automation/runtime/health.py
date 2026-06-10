"""automation.runtime.health — /healthz heartbeat + a tiny HTTP liveness server.

Railway (and any process supervisor) can detect a CRASHED process by exit code,
but NOT a STUCK one — a daemon whose tick loop has wedged keeps the process
alive while doing no work.  This module closes that gap:

* ``Heartbeat`` — a thread-safe last-tick timestamp.  ``run_once`` calls
  ``beat(now_ts)`` every tick; ``is_healthy(now_ts)`` reports stale once no beat
  has landed within ``max_silence_seconds``.
* ``start_health_server`` — a daemon-thread HTTP server exposing ``GET /healthz``
  → ``200 ok`` when the heartbeat is fresh, ``503 stale`` when it is not (or has
  never beaten).  A liveness probe that returns 503 lets the supervisor restart a
  wedged daemon.
"""

from __future__ import annotations

import datetime
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer


class Heartbeat:
    """A thread-safe last-tick timestamp with a staleness window."""

    def __init__(self, max_silence_seconds: int = 180) -> None:
        self._max = max_silence_seconds
        self._last_ts: str | None = None

    def beat(self, now_ts: str) -> None:
        """Record ``now_ts`` (ISO 8601) as the most recent tick."""
        self._last_ts = now_ts

    def is_healthy(self, now_ts: str) -> bool:
        """True iff a beat landed within ``max_silence_seconds`` of ``now_ts``.

        Returns ``False`` before the first beat (never-started ⇒ not live).
        """
        if self._last_ts is None:
            return False
        last = datetime.datetime.fromisoformat(self._last_ts)
        now = datetime.datetime.fromisoformat(now_ts)
        return (now - last).total_seconds() <= self._max


def start_health_server(
    heartbeat: Heartbeat,
    *,
    port: int,
    clock: Callable[[], str],
) -> HTTPServer:
    """Start a daemon-thread ``GET /healthz`` liveness server and return it.

    ``port=0`` lets the OS pick a free port (read it from
    ``server.server_address[1]``).  Binds ``0.0.0.0`` for Railway.  The returned
    server runs on a background daemon thread; call ``server.shutdown()`` to stop.
    """

    class _H(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler dispatch name
            if self.path != "/healthz":
                self.send_response(404)
                self.end_headers()
                return
            ok = heartbeat.is_healthy(clock())
            self.send_response(200 if ok else 503)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok" if ok else b"stale")

        def log_message(self, *_a: object) -> None:
            return

    srv = HTTPServer(("0.0.0.0", port), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
