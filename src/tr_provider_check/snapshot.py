"""Access the single packaged production contract snapshot."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def load_contract() -> dict[str, Any]:
    snapshot_file = resources.files("tr_provider_check").joinpath(
        "data/contract_snapshot.json"
    )
    payload = json.loads(snapshot_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("contract snapshot root must be an object")
    return payload


contract_version = str(load_contract()["contract_version"])
