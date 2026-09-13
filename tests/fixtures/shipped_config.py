"""Load local JSON or the chart's rendered runtime ConfigMap."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

CHART = Path(__file__).resolve().parents[2] / "charts" / "cogniverse"


def load_shipped_config(path: Path) -> dict:
    if path != CHART / "files" / "config.json":
        return json.loads(path.read_text(encoding="utf-8"))
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "cogniverse",
            str(CHART),
            "--show-only",
            "templates/configmap.yaml",
            "--set",
            "runtime.qualityMonitor.tenantId=test-tenant",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    configmap = next(
        doc
        for doc in yaml.safe_load_all(rendered.stdout)
        if doc["metadata"]["name"] == "cogniverse-config"
    )
    return json.loads(configmap["data"]["config.json"])
