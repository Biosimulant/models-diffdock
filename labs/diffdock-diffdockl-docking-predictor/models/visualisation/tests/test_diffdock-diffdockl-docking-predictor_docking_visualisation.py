from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


_MODEL_DIR = Path(__file__).resolve().parents[1]


def _find_bsim_src(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        for candidate in (parent / "biosim" / "src", parent / "bsim-active" / "biosim" / "src"):
            if (candidate / "biosim").is_dir():
                return candidate
    return None


_BSIM_SRC = _find_bsim_src(_MODEL_DIR)
if _BSIM_SRC is not None and str(_BSIM_SRC) not in sys.path:
    sys.path.insert(0, str(_BSIM_SRC))

from biosim.signals import RecordSignal


def _load_model():
    manifest = yaml.safe_load((_MODEL_DIR / "model.yaml").read_text())
    entrypoint = manifest["biosim"]["entrypoint"]
    module_name, class_name = entrypoint.split(":")
    module_rel = Path(*module_name.split(".")).with_suffix(".py")
    wrapper_path = _MODEL_DIR / module_rel
    unique_name = f"done_visualisation__{_MODEL_DIR.parent.parent.name}"
    spec = importlib.util.spec_from_file_location(unique_name, wrapper_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    module_cls = getattr(module, class_name)
    return module_cls(**dict(manifest["biosim"].get("init_kwargs", {}))), float(manifest["biosim"]["communication_step"])


def test_visualisation_model_renders_docking_visuals(tmp_path):
    module, step = _load_model()
    alias = module.source_alias
    mode = module.mode

    def _record(name: str, payload: object) -> RecordSignal:
        spec = module.inputs()[name]
        return RecordSignal(source="test", name=name, value={"payload": payload}, emitted_at=step, spec=spec)

    completed = _record(f"{alias}_run_metadata", {"status": "completed"})

    if mode == "vina":
        structure_path = tmp_path / "top_complex.pdb"
        structure_path.write_text("ATOM\nEND\n", encoding="utf-8")
        inputs = {
            f"{alias}_run_metadata": completed,
            f"{alias}_structure_artifacts": _record(f"{alias}_structure_artifacts", {"top_complex_file": str(structure_path)}),
            f"{alias}_docking_summary": _record(f"{alias}_docking_summary", {"top_pose_affinity_kcal_mol": -9.1, "scoring": "vina", "pose_count": 2}),
            f"{alias}_pose_summary": _record(f"{alias}_pose_summary", [{"rank": 1, "affinity_kcal_mol": -9.1, "rmsd_lb": 0.0, "rmsd_ub": 0.0, "pose_file": str(tmp_path / 'pose1.pdbqt')}]),
        }
    elif mode == "boltz":
        structure_path = tmp_path / "complex.cif"
        structure_path.write_text("data_test\n#\n", encoding="utf-8")
        inputs = {
            f"{alias}_run_metadata": completed,
            f"{alias}_structure_artifacts": _record(f"{alias}_structure_artifacts", {"structure_file": str(structure_path)}),
            f"{alias}_confidence_summary": _record(f"{alias}_confidence_summary", {"mean_plddt": 91.2, "mean_iptm": 0.72, "mean_ligand_iptm": 0.68}),
            f"{alias}_affinity_summary": _record(f"{alias}_affinity_summary", {"predicted_affinity_value": -8.4, "predicted_affinity_unit": "kcal/mol", "affinity_probability_binary": 0.81}),
        }
    else:
        structure_path = tmp_path / "top_complex.pdb"
        structure_path.write_text("ATOM\nEND\n", encoding="utf-8")
        inputs = {
            f"{alias}_run_metadata": completed,
            f"{alias}_structure_artifacts": _record(f"{alias}_structure_artifacts", {"top_complex_file": str(structure_path)}),
            f"{alias}_confidence_summary": _record(f"{alias}_confidence_summary", {"top_pose_confidence": 0.77, "confidence_band": "high", "pose_count": 2}),
            f"{alias}_pose_summary": _record(f"{alias}_pose_summary", [{"rank": 1, "confidence": 0.77, "confidence_band": "high", "file_path": str(tmp_path / 'pose1.sdf')}]),
        }

    module.set_inputs(inputs)
    module.advance_window(0.0, step)

    assert module.outputs() == {}
    assert module.get_outputs() == {}
    visuals = module.visualize()
    assert isinstance(visuals, list) and len(visuals) == 2
    assert [visual["render"] for visual in visuals] == ["structure3d", "table"]
