"""Shared fixtures for the sim-runtime tests.

``ServerMetadata.__post_init__`` hits ``ipinfo.io`` for a geolocation string,
which adds a multi-second delay (and occasional network failures) to every
``SimRuntime.add_robot`` call. Patch it out for all tests in this package.
"""

from __future__ import annotations

import pytest
from openpi_client.schemas import ServerMetadata


@pytest.fixture(autouse=True)
def _skip_server_metadata_geolookup(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(self: ServerMetadata) -> None:
        self.location = "sim"

    monkeypatch.setattr(ServerMetadata, "__post_init__", _stub)
