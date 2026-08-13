from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.mockserver.app import MockOpenAIServer


@pytest.fixture(scope="session")
def _session_mock_server() -> Iterator[MockOpenAIServer]:
    server = MockOpenAIServer()
    server.start()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def mock_server(_session_mock_server: MockOpenAIServer) -> Iterator[MockOpenAIServer]:
    _session_mock_server.reset()
    yield _session_mock_server
    _session_mock_server.reset()
