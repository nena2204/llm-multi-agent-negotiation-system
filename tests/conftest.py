import os
import socket

import pytest


@pytest.fixture(autouse=True)
def block_network_in_default_tests(request, monkeypatch):
    live_enabled = (
        request.node.get_closest_marker("live_api") is not None
        and os.environ.get("RUN_LIVE_LLM_TESTS") == "1"
    )
    if live_enabled:
        return

    def blocked(*_args, **_kwargs):
        raise AssertionError(
            "network access is disabled in the default test suite; use the live_api opt-in"
        )

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)

