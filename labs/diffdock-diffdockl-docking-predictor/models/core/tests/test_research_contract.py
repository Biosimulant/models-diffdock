"""Regression tests for controls, reproducibility and unusable result rejection."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml
from biosim.signals import make_signal
from src.diffdockl_docking_predictor import DiffDockLDockingPredictor


def test_partial_options_preserve_lab_preset():
    model = DiffDockLDockingPredictor(default_run_options={"samples_per_complex": 2, "inference_steps": 4, "batch_size": 2})
    model.set_inputs({"run_options": make_signal(source="test", name="run_options", value={"complex_name": "trial"}, emitted_at=0, spec=None)})
    options = model._resolved_options()
    assert (options["samples_per_complex"], options["inference_steps"], options["batch_size"], options["complex_name"]) == (2, 4, 2, "trial")


@pytest.mark.parametrize("options", [True, [], "samples=2", {1: 2}])
def test_invalid_options_container_rejected(options):
    with pytest.raises(ValueError):
        DiffDockLDockingPredictor(default_run_options=options)


@pytest.mark.parametrize("key", ["samples_per_complex", "inference_steps", "batch_size"])
@pytest.mark.parametrize("value", [True, 0, -1, 2.5, float("nan"), float("inf")])
def test_invalid_integer_controls_rejected(key, value):
    model = DiffDockLDockingPredictor(default_run_options={key: value})
    with pytest.raises(ValueError):
        model._resolved_options()


@pytest.mark.parametrize("name", ["../outside", "/tmp/escape", "a/b", "..", "", "x" * 101])
def test_output_name_cannot_escape_run_directory(name):
    with pytest.raises(ValueError):
        DiffDockLDockingPredictor(default_run_options={"complex_name": name})._resolved_options()


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan")])
def test_timeout_must_be_bounded(value):
    with pytest.raises(ValueError):
        DiffDockLDockingPredictor(command_timeout_s=value)


def test_stereochemical_smiles_are_not_paths():
    model = DiffDockLDockingPredictor()
    for smiles in ["C/C=C/C", "C/C=C\\C"]:
        assert model._resolve_ligand_description(smiles) == {"kind": "smiles", "value": smiles}


def test_effective_config_honors_controls_under_upstream_yaml_precedence(tmp_path):
    model = DiffDockLDockingPredictor(default_run_options={"samples_per_complex": 2, "inference_steps": 4, "batch_size": 2})
    repo = tmp_path / "repo"
    repo.mkdir()
    # These conflicting defaults reproduce upstream v1.1.3's post-argparse merge.
    (repo / "default_inference_args.yaml").write_text("samples_per_complex: 10\ninference_steps: 20\nactual_steps: 19\nno_final_step_noise: true\nmodel_dir: ./old\n")
    command = model._build_command(python_executable=sys.executable, repo_dir=repo, output_dir=tmp_path / "run/output", protein_path="receptor.pdb", ligand_input={"kind": "smiles", "value": "CC"}, options=model._resolved_options())
    config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
    assert config["samples_per_complex"] == 2
    assert config["inference_steps"] == config["actual_steps"] == 4
    assert config["no_final_step_noise"] is True
    assert config["model_dir"] == str(repo / "workdir/v1.1/score_model")
    assert config["ligand_description"] == "CC"


@pytest.mark.parametrize("names", [
    ["rank1_confidencenan.sdf"], ["rank1_confidenceinf.sdf"],
    ["rank0_confidence0.5.sdf"], ["rank2_confidence0.5.sdf"],
    ["rank1.sdf"], ["rank1_confidence0.5.sdf", "rank1_confidence0.6.sdf"],
])
def test_unusable_rank_sets_rejected(tmp_path, names):
    for name in names:
        (tmp_path / name).write_text("fixture")
    with pytest.raises(ValueError):
        DiffDockLDockingPredictor()._collect_pose_records(tmp_path)


def test_ranked_pose_used_for_both_summary_and_visualization(tmp_path):
    (tmp_path / "rank1.sdf").write_text("unranked alias")
    ranked = tmp_path / "rank1_confidence0.25.sdf"
    ranked.write_text("ranked pose")
    rows, top, _ = DiffDockLDockingPredictor()._collect_pose_records(tmp_path)
    assert str(top) == rows[0]["file_path"] == str(ranked)


def test_wrong_complex_output_is_not_silently_adopted(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "rank1_confidence0.5.sdf").write_text("fixture")
    with pytest.raises(FileNotFoundError):
        DiffDockLDockingPredictor()._find_prediction_dir(tmp_path, "requested")


def test_cached_checkout_revision_and_dirty_state_are_checked(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "tracked"
    tracked.write_text("original")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    metadata = {"runtime_setup_commands": []}
    model = DiffDockLDockingPredictor(diffdock_git_commit=revision)
    model._verify_repo_checkout(tmp_path, metadata)
    assert metadata["upstream_commit"] == revision
    tracked.write_text("changed")
    with pytest.raises(RuntimeError, match="modified tracked"):
        model._verify_repo_checkout(tmp_path, metadata)
    with pytest.raises(RuntimeError, match="expected"):
        DiffDockLDockingPredictor()._verify_repo_checkout(tmp_path, metadata)


def test_managed_runtime_rejects_unsupported_host_before_install(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux and Python 3.11"):
        DiffDockLDockingPredictor()._validate_runtime_platform()


def checkpoint_fixture(tmp_path):
    archive = tmp_path / "bundle.zip"
    payload = b"test checkpoint bytes, not scientific weights"
    name = "score_model/example.pt"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, payload)
    manifest = {"source": archive.as_uri(), "bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "files": {name: hashlib.sha256(payload).hexdigest()}}
    root = tmp_path / "model"
    (root / "data").mkdir(parents=True)
    (root / "data/diffdock-checkpoints.json").write_text(json.dumps(manifest))
    model = DiffDockLDockingPredictor()
    model.model_root = root
    return model, archive, name


def test_checkpoint_download_verified_and_modified_cache_rejected(tmp_path):
    model, archive, name = checkpoint_fixture(tmp_path)
    repo = tmp_path / "repo"
    metadata = {}
    model._ensure_checkpoints(repo, metadata)
    archive.unlink()  # A fully verified cache needs no download.
    model._ensure_checkpoints(repo, metadata)
    (repo / "workdir/v1.1" / name).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="cached DiffDock checkpoint mismatch"):
        model._ensure_checkpoints(repo, metadata)


def test_corrupt_checkpoint_archive_is_not_installed(tmp_path):
    model, archive, name = checkpoint_fixture(tmp_path)
    archive.write_bytes(b"bad download")
    repo = tmp_path / "repo"
    with pytest.raises(RuntimeError, match="checksum verification"):
        model._ensure_checkpoints(repo, {})
    assert not (repo / "workdir/v1.1" / name).exists()


def test_confidence_is_explicitly_not_affinity_or_probability():
    summary = DiffDockLDockingPredictor()._build_confidence_summary([{"rank": 1, "confidence": 0.3, "confidence_band": "high"}])
    assert "not a binding affinity or calibrated probability" in summary["interpretation"]
