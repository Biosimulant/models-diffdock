# SPDX-FileCopyrightText: 2026-present Biosimulant Team
# SPDX-License-Identifier: Apache-2.0
"""Dedicated visualisation model for docking labs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from biosim import BioModule
from biosim.signals import AcceptedSignalProfile, BioSignal, SignalSpec


def _signal_value(signal: BioSignal | None) -> Any:
    if signal is None:
        return None
    value = getattr(signal, "value", None)
    if isinstance(value, dict) and set(value.keys()) == {"payload"}:
        return value["payload"]
    return value


def _record_input_spec(description: str) -> SignalSpec:
    return SignalSpec.record(
        schema={"payload": "json"},
        accepted_profiles=(
            AcceptedSignalProfile(signal_type="record", schema={"payload": "json"}),
            AcceptedSignalProfile(signal_type="scalar"),
        ),
        description=description,
    )


class DockingVisualisationModel(BioModule):
    def __init__(
        self,
        integration_step: float = 0.01,
        source_alias: str = "core",
        mode: str = "vina",
        lab_title: str = "Docking Lab",
    ) -> None:
        self.integration_step = float(integration_step)
        self.source_alias = source_alias
        self.mode = mode
        self.lab_title = lab_title
        self._inputs: Dict[str, BioSignal] = {}

    def inputs(self) -> dict[str, SignalSpec]:
        names_by_mode = {
            "vina": ["pose_summary", "docking_summary", "structure_artifacts", "run_metadata"],
            "boltz": ["affinity_summary", "confidence_summary", "structure_artifacts", "run_metadata"],
            "diffdock": ["pose_summary", "confidence_summary", "structure_artifacts", "run_metadata"],
        }
        names = names_by_mode[self.mode]
        return {
            f"{self.source_alias}_{name}": _record_input_spec(f"Internal {name} input from the sibling core model.")
            for name in names
        }

    def outputs(self) -> dict[str, SignalSpec]:
        return {}

    def reset(self) -> None:
        self._inputs = {}

    def set_inputs(self, signals: dict[str, BioSignal]) -> None:
        self._inputs.update(signals or {})

    def advance_window(self, start: float | None = None, end: float | None = None, inputs: dict[str, BioSignal] | None = None) -> dict[str, BioSignal]:
        if inputs:
            self.set_inputs(inputs)
        return {}

    def get_outputs(self) -> dict[str, BioSignal]:
        return {}

    def visualize(self) -> Optional[list[dict[str, Any]]]:
        if self.mode == "vina":
            return self._visualize_vina()
        if self.mode == "boltz":
            return self._visualize_boltz()
        return self._visualize_diffdock()

    def _input_value(self, name: str) -> Any:
        return _signal_value(self._inputs.get(f"{self.source_alias}_{name}"))

    def _visualize_vina(self) -> Optional[list[dict[str, Any]]]:
        run_metadata = self._input_value("run_metadata")
        artifacts = self._input_value("structure_artifacts")
        docking_summary = self._input_value("docking_summary")
        poses = self._input_value("pose_summary")
        if not isinstance(run_metadata, Mapping) or run_metadata.get("status") != "completed":
            return None
        if not isinstance(artifacts, Mapping) or not isinstance(docking_summary, Mapping) or not isinstance(poses, list):
            return None
        top_complex_path = self._resolved_path(artifacts.get("top_complex_file"))
        if top_complex_path is None:
            return None
        rows = []
        for row in poses:
            if not isinstance(row, Mapping):
                continue
            rows.append([
                str(row.get("rank", "")),
                "" if row.get("affinity_kcal_mol") is None else str(row.get("affinity_kcal_mol")),
                "" if row.get("rmsd_lb") is None else str(row.get("rmsd_lb")),
                "" if row.get("rmsd_ub") is None else str(row.get("rmsd_ub")),
                Path(str(row.get("pose_file", ""))).name,
            ])
        return [
            {
                "render": "structure3d",
                "description": "Top-ranked AutoDock Vina complex for the latest docking run.",
                "data": {
                    "title": "Top-Ranked Docked Complex",
                    "source": {"kind": "artifact", "artifact_id": self._artifact_id(top_complex_path), "path": str(top_complex_path)},
                    "format": "pdb",
                    "annotations": [
                        {"label": "Top Pose Affinity (kcal/mol)", "value": docking_summary.get("top_pose_affinity_kcal_mol")},
                        {"label": "Scoring", "value": docking_summary.get("scoring")},
                        {"label": "Pose Count", "value": docking_summary.get("pose_count")},
                    ],
                    "initial_view": {"reset_camera": True},
                },
            },
            {
                "render": "table",
                "description": "Ranked pose summary from the latest AutoDock Vina run.",
                "data": {
                    "title": "AutoDock Vina Pose Summary",
                    "columns": ["Rank", "Affinity", "RMSD l.b.", "RMSD u.b.", "Pose File"],
                    "rows": rows,
                },
            },
        ]

    def _visualize_boltz(self) -> Optional[list[dict[str, Any]]]:
        run_metadata = self._input_value("run_metadata")
        artifacts = self._input_value("structure_artifacts")
        confidence = self._input_value("confidence_summary")
        affinity = self._input_value("affinity_summary")
        if not isinstance(run_metadata, Mapping) or run_metadata.get("status") != "completed":
            return None
        if not isinstance(artifacts, Mapping):
            return None
        structure_path = self._resolved_path(artifacts.get("structure_file"))
        if structure_path is None:
            return None
        structure_format = self._structure_format(structure_path)
        if structure_format is None:
            return None
        annotations = self._build_boltz_annotations(confidence, affinity)
        return [
            {
                "render": "structure3d",
                "description": "Top-ranked Boltz structure prediction for the latest protein-ligand run.",
                "data": {
                    "title": "Predicted Complex Structure",
                    "source": {"kind": "artifact", "artifact_id": self._artifact_id(structure_path), "path": str(structure_path)},
                    "format": structure_format,
                    "annotations": [{"label": label, "value": value} for label, value in annotations],
                    "initial_view": {"reset_camera": True},
                },
            },
            {
                "render": "table",
                "description": "Key affinity and confidence metrics extracted from the latest Boltz outputs.",
                "data": {"title": "Boltz Summary", "columns": ["Metric", "Value"], "rows": [[label, str(value)] for label, value in annotations]},
            },
        ]

    def _visualize_diffdock(self) -> Optional[list[dict[str, Any]]]:
        run_metadata = self._input_value("run_metadata")
        artifacts = self._input_value("structure_artifacts")
        confidence = self._input_value("confidence_summary")
        poses = self._input_value("pose_summary")
        if not isinstance(run_metadata, Mapping) or run_metadata.get("status") != "completed":
            return None
        if not isinstance(artifacts, Mapping) or not isinstance(poses, list):
            return None
        top_complex_path = self._resolved_path(artifacts.get("top_complex_file"))
        if top_complex_path is None:
            return None
        rows = []
        for row in poses:
            if not isinstance(row, Mapping):
                continue
            rows.append([
                str(row.get("rank", "")),
                "" if row.get("confidence") is None else str(row.get("confidence")),
                str(row.get("confidence_band") or ""),
                Path(str(row.get("file_path", ""))).name,
            ])
        return [
            {
                "render": "structure3d",
                "description": "Top-ranked DiffDock-L complex for the latest docking run.",
                "data": {
                    "title": "Top-Ranked Docked Complex",
                    "source": {"kind": "artifact", "artifact_id": self._artifact_id(top_complex_path), "path": str(top_complex_path)},
                    "format": "pdb",
                    "annotations": [
                        {"label": "Top Pose Confidence", "value": confidence.get("top_pose_confidence") if isinstance(confidence, Mapping) else None},
                        {"label": "Confidence Band", "value": confidence.get("confidence_band") if isinstance(confidence, Mapping) else None},
                        {"label": "Pose Count", "value": confidence.get("pose_count") if isinstance(confidence, Mapping) else None},
                    ],
                    "initial_view": {"reset_camera": True},
                },
            },
            {
                "render": "table",
                "description": "Ranked pose summary from the latest DiffDock-L run.",
                "data": {"title": "DiffDock Pose Summary", "columns": ["Rank", "Confidence", "Band", "Pose File"], "rows": rows},
            },
        ]

    @staticmethod
    def _artifact_id(path: Path) -> str:
        return hashlib.sha1(str(path).encode("utf-8")).hexdigest()

    @staticmethod
    def _resolved_path(value: Any) -> Optional[Path]:
        if not isinstance(value, str) or not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    @staticmethod
    def _structure_format(path: Path) -> Optional[str]:
        suffix = path.suffix.lower()
        if suffix in {".cif", ".mmcif"}:
            return "mmcif"
        if suffix in {".pdb", ".ent"}:
            return "pdb"
        return None

    @staticmethod
    def _build_boltz_annotations(confidence: Any, affinity: Any) -> list[tuple[str, Any]]:
        confidence = confidence if isinstance(confidence, Mapping) else {}
        affinity = affinity if isinstance(affinity, Mapping) else {}
        return [
            ("Predicted affinity", affinity.get("predicted_affinity_value")),
            ("Affinity unit", affinity.get("predicted_affinity_unit")),
            ("Affinity probability", affinity.get("affinity_probability_binary")),
            ("Mean pLDDT", confidence.get("mean_plddt")),
            ("Mean ipTM", confidence.get("mean_iptm")),
            ("Mean ligand ipTM", confidence.get("mean_ligand_iptm")),
        ]
