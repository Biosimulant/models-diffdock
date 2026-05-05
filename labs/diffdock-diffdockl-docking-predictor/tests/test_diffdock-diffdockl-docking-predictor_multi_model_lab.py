from __future__ import annotations

from pathlib import Path

import yaml


def test_lab_manifest_uses_embedded_visualisation_model():
    lab_dir = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((lab_dir / "lab.yaml").read_text())
    model_aliases = [entry["alias"] for entry in manifest["models"]]
    assert "visualisation" in model_aliases
    assert all(str(entry["path"]).startswith("models/") for entry in manifest["models"])

    assert not any(str(entry["maps_to"]).startswith("visualisation.") for entry in manifest["io"]["outputs"])

    wiring_targets = {target for entry in manifest["wiring"] for target in entry["to"]}
    assert any(target.startswith("visualisation.") for target in wiring_targets)
