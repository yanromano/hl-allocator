"""Tests for automation.runtime.health (Heartbeat + /healthz liveness server).

Written before the implementation (TDD RED → GREEN).
"""

from __future__ import annotations

import urllib.error
import urllib.request


def test_heartbeat_healthy_within_window() -> None:
    from automation.runtime.health import Heartbeat

    hb = Heartbeat(max_silence_seconds=180)
    hb.beat(now_ts="2026-05-18T00:00:00+00:00")
    assert hb.is_healthy(now_ts="2026-05-18T00:02:00+00:00") is True


def test_heartbeat_unhealthy_after_window() -> None:
    from automation.runtime.health import Heartbeat

    hb = Heartbeat(max_silence_seconds=180)
    hb.beat(now_ts="2026-05-18T00:00:00+00:00")
    assert hb.is_healthy(now_ts="2026-05-18T00:05:00+00:00") is False


def test_heartbeat_unhealthy_before_first_beat() -> None:
    from automation.runtime.health import Heartbeat

    assert Heartbeat(max_silence_seconds=180).is_healthy(now_ts="2026-05-18T00:00:00+00:00") is False


def test_health_server_serves_200_after_beat_503_before() -> None:
    # hermetic localhost integration: start server, GET /healthz, assert 503 then beat -> 200
    from automation.runtime.health import Heartbeat, start_health_server

    hb = Heartbeat(max_silence_seconds=180)
    clock = {"ts": "2026-05-18T00:00:00+00:00"}
    srv = start_health_server(hb, port=0, clock=lambda: clock["ts"])  # port 0 -> OS picks
    try:
        port = srv.server_address[1]

        def get() -> int:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                return e.code

        assert get() == 503  # no beat yet
        hb.beat(now_ts=clock["ts"])
        assert get() == 200
    finally:
        srv.shutdown()
