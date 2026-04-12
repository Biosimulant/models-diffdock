from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import yaml


def _set_required_inputs(module, BioSignal, *, protein_path: str | None = None, ligand_description: str | None = None, run_options: dict | None = None):
    signals = {}
    if protein_path is not None:
        signals["protein_path"] = BioSignal(source="test", name="protein_path", value=protein_path, time=0.0)
    if ligand_description is not None:
        signals["ligand_description"] = BioSignal(source="test", name="ligand_description", value=ligand_description, time=0.0)
    if run_options is not None:
        signals["run_options"] = BioSignal(source="test", name="run_options", value=run_options, time=0.0)
    module.set_inputs(signals)


def _patch_invoke_command(monkeypatch, predictor_cls, handler):
    def fake_invoke(self, command, *, cwd, timeout, env, phase):
        return handler(
            [str(item) for item in command],
            cwd=cwd,
            timeout=timeout,
            env=env,
            phase=phase,
        )

    monkeypatch.setattr(predictor_cls, "_invoke_command", fake_invoke)


def test_instantiation(biosim, tmp_path):
    from src.diffdockl_docking_predictor import DiffDockLDockingPredictor

    module = DiffDockLDockingPredictor(work_dir=str(tmp_path))
    assert module.min_dt > 0
    assert module.runtime_mode == "managed"
    assert module.inputs() == {"protein_path", "ligand_description", "run_options"}
    assert module.outputs() == {"pose_summary", "confidence_summary", "structure_artifacts", "run_metadata"}
    assert module.requirements_file.name == "runtime-gpu.txt"


def test_missing_inputs_surface_error_metadata(biosim, tmp_path):
    from src.diffdockl_docking_predictor import DiffDockLDockingPredictor

    module = DiffDockLDockingPredictor(work_dir=str(tmp_path))
    module.advance_to(0.1)

    outputs = module.get_outputs()
    assert outputs["run_metadata"].value["status"] == "error"
    assert "protein_path" in outputs["run_metadata"].value["error"]
    assert module.visualize() is None


def test_model_relative_path_resolution_uses_checked_in_asset(biosim):
    from src.diffdockl_docking_predictor import DiffDockLDockingPredictor

    module = DiffDockLDockingPredictor()
    resolved = module._resolve_required_input_path("data/1a0q/1a0q_protein_processed.pdb", input_name="protein_path")
    assert Path(resolved).is_file()
    assert Path(resolved).name == "1a0q_protein_processed.pdb"


def test_model_relative_ligand_file_resolution_uses_checked_in_asset(biosim):
    from src.diffdockl_docking_predictor import DiffDockLDockingPredictor

    module = DiffDockLDockingPredictor()
    resolved = module._resolve_ligand_description("data/1a0q/1a0q_ligand.sdf")
    assert resolved["kind"] == "path"
    assert Path(resolved["value"]).is_file()
    assert Path(resolved["value"]).name == "1a0q_ligand.sdf"


def test_run_options_validation_rejects_unknown_keys(biosim, tmp_path):
    from src.diffdockl_docking_predictor import DiffDockLDockingPredictor
    from biosim.signals import BioSignal

    module = DiffDockLDockingPredictor(work_dir=str(tmp_path))
    _set_required_inputs(
        module,
        BioSignal,
        protein_path="data/1a0q/1a0q_protein_processed.pdb",
        ligand_description="COc(cc1)ccc1C#N",
        run_options={"not_supported": 1},
    )
    module.advance_to(0.1)

    metadata = module.get_outputs()["run_metadata"].value
    assert metadata["status"] == "error"
    assert "unsupported run_options key" in metadata["error"]


def test_managed_runtime_bootstraps_and_parses_outputs(biosim, tmp_path, monkeypatch):
    from src.diffdockl_docking_predictor import DiffDockLDockingPredictor
    from biosim.signals import BioSignal

    commands: list[list[str]] = []

    def fake_invoke(command, *, cwd, timeout, env, phase):  # noqa: ARG001
        commands.append(command)
        if command[:4] == ["git", "clone", "--branch", "v1.1.3"]:
            repo_dir = Path(command[-1])
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / ".git").mkdir()
            (repo_dir / "workdir" / "v1.1").mkdir(parents=True, exist_ok=True)
            (repo_dir / "default_inference_args.yaml").write_text("samples_per_complex: 10\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="cloned", stderr="")
        if "-m" in command and "venv" in command:
            venv_root = Path(command[-1])
            (venv_root / "bin").mkdir(parents=True, exist_ok=True)
            (venv_root / "bin" / "python").write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="venv", stderr="")
        if "-m" in command and "pip" in command and "install" in command:
            return subprocess.CompletedProcess(command, 0, stdout="pip", stderr="")
        if command[1:3] == ["-m", "inference"]:
            out_dir = Path(command[command.index("--out_dir") + 1])
            complex_name = command[command.index("--complex_name") + 1]
            prediction_dir = out_dir / complex_name
            prediction_dir.mkdir(parents=True, exist_ok=True)
            (prediction_dir / "rank1.sdf").write_text("rank1", encoding="utf-8")
            (prediction_dir / "rank1_confidence0.72.sdf").write_text("rank1-c", encoding="utf-8")
            (prediction_dir / "rank2_confidence-0.45.sdf").write_text("rank2-c", encoding="utf-8")
            (prediction_dir / "rank1_reverseprocess.pdb").write_text("MODEL\nENDMDL\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    _patch_invoke_command(monkeypatch, DiffDockLDockingPredictor, fake_invoke)
    monkeypatch.setattr(
        DiffDockLDockingPredictor,
        "_ligand_pdb_lines_from_sdf",
        lambda self, _path, _python_executable=None: [
            "HETATM    1   C1 LIG Z   1      11.111  22.222  33.333  1.00  0.00           C"
        ],
    )

    module = DiffDockLDockingPredictor(
        work_dir=str(tmp_path),
        runtime_dir=str(tmp_path / "runtime"),
    )
    _set_required_inputs(
        module,
        BioSignal,
        protein_path="data/1a0q/1a0q_protein_processed.pdb",
        ligand_description="COc(cc1)ccc1C#N",
        run_options={
            "complex_name": "1a0q_custom",
            "samples_per_complex": 2,
            "inference_steps": 4,
            "batch_size": 2,
            "save_visualisation": True,
        },
    )
    module.advance_to(0.5)

    outputs = module.get_outputs()
    metadata = outputs["run_metadata"].value
    pose_summary = outputs["pose_summary"].value
    confidence = outputs["confidence_summary"].value
    artifacts = outputs["structure_artifacts"].value

    assert metadata["status"] == "completed"
    assert metadata["repo_bootstrapped"] is True
    assert metadata["runtime_bootstrapped"] is True
    assert metadata["command"][1:3] == ["-m", "inference"]
    assert confidence["top_pose_confidence"] == 0.72
    assert confidence["confidence_band"] == "high"
    assert len(pose_summary) == 2
    assert pose_summary[0]["rank"] == 1
    assert pose_summary[0]["confidence_band"] == "high"
    assert Path(artifacts["top_complex_file"]).is_file()
    assert Path(artifacts["top_pose_file"]).is_file()
    assert Path(artifacts["confidence_file"]).is_file()
    assert Path(artifacts["pose_summary_file"]).is_file()
    assert Path(artifacts["rank_1_reverseprocess_file"]).is_file()
    assert any(cmd[:4] == ["git", "clone", "--branch", "v1.1.3"] for cmd in commands)
    assert any("--save_visualisation" in cmd for cmd in commands if cmd and cmd[0].endswith("python"))

    visuals = module.visualize()
    assert visuals is not None
    assert visuals[0]["render"] == "structure3d"
    assert visuals[0]["data"]["format"] == "pdb"
    assert visuals[0]["data"]["source"]["kind"] == "artifact"
    assert visuals[1]["render"] == "table"
    assert visuals[1]["data"]["columns"] == ["Rank", "Confidence", "Band", "Pose File"]


def test_generated_structure_paths_remain_absolute_without_canonicalizing(biosim, tmp_path):
    from src.diffdockl_docking_predictor import DiffDockLDockingPredictor

    module = DiffDockLDockingPredictor(work_dir=str(tmp_path), runtime_dir=str(tmp_path / "runtime"))

    prediction_dir = tmp_path / "runs" / ".." / "artifacts" / "request"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    top_pose_path = prediction_dir / "rank1_confidence0.16.sdf"
    top_pose_path.write_text("pose", encoding="utf-8")
    top_complex_file = prediction_dir / "top_rank_complex.pdb"
    top_complex_file.write_text("HEADER TEST\n", encoding="utf-8")
    confidence_file = prediction_dir / "confidence_summary.json"
    confidence_file.write_text("{}", encoding="utf-8")
    pose_summary_file = prediction_dir / "pose_summary.json"
    pose_summary_file.write_text("[]", encoding="utf-8")
    reverseprocess_file = prediction_dir / "rank1_reverseprocess.pdb"
    reverseprocess_file.write_text("MODEL\nENDMDL\n", encoding="utf-8")

    artifacts = module._build_structure_artifacts(
        prediction_dir=prediction_dir,
        pose_records=[{"rank": 1, "file_path": str(top_pose_path), "confidence": 0.16, "confidence_band": "high"}],
        top_pose_path=top_pose_path,
        top_complex_file=top_complex_file,
        confidence_file=confidence_file,
        pose_summary_file=pose_summary_file,
        reverseprocess_files={1: reverseprocess_file},
    )

    assert artifacts["prediction_dir"] == str(prediction_dir)
    assert artifacts["top_complex_file"] == str(top_complex_file)
    assert artifacts["rank_1_file"] == str(top_pose_path)
    assert "/../" in artifacts["prediction_dir"]

    module._cached_payloads = {
        "run_metadata": {"status": "completed"},
        "structure_artifacts": artifacts,
        "confidence_summary": {"top_pose_confidence": 0.16, "confidence_band": "high", "pose_count": 1},
        "pose_summary": [{"rank": 1, "confidence": 0.16, "confidence_band": "high", "file_path": str(top_pose_path)}],
    }
    visuals = module.visualize()
    assert visuals is not None
    assert visuals[0]["data"]["source"]["path"] == str(top_complex_file)
    assert "/../" in visuals[0]["data"]["source"]["path"]


def test_advance_emits_progress_events_for_long_steps(biosim, tmp_path, monkeypatch, capsys):
    from src.diffdockl_docking_predictor import DiffDockLDockingPredictor
    from biosim.signals import BioSignal

    def fake_invoke(command, *, cwd, timeout, env, phase):  # noqa: ARG001
        if command[:4] == ["git", "clone", "--branch", "v1.1.3"]:
            repo_dir = Path(command[-1])
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / ".git").mkdir()
            (repo_dir / "workdir" / "v1.1").mkdir(parents=True, exist_ok=True)
            (repo_dir / "default_inference_args.yaml").write_text("samples_per_complex: 10\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="cloned", stderr="")
        if "-m" in command and "venv" in command:
            venv_root = Path(command[-1])
            (venv_root / "bin").mkdir(parents=True, exist_ok=True)
            (venv_root / "bin" / "python").write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="venv", stderr="")
        if "-m" in command and "pip" in command and "install" in command:
            return subprocess.CompletedProcess(command, 0, stdout="pip", stderr="")
        if command[1:3] == ["-m", "inference"]:
            time.sleep(0.03)
            out_dir = Path(command[command.index("--out_dir") + 1])
            complex_name = command[command.index("--complex_name") + 1]
            prediction_dir = out_dir / complex_name
            prediction_dir.mkdir(parents=True, exist_ok=True)
            (prediction_dir / "rank1.sdf").write_text("rank1", encoding="utf-8")
            (prediction_dir / "rank1_confidence0.72.sdf").write_text("rank1-c", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    _patch_invoke_command(monkeypatch, DiffDockLDockingPredictor, fake_invoke)
    monkeypatch.setattr(
        DiffDockLDockingPredictor,
        "_ligand_pdb_lines_from_sdf",
        lambda self, _path, _python_executable=None: [
            "HETATM    1   C1 LIG Z   1      11.111  22.222  33.333  1.00  0.00           C"
        ],
    )

    module = DiffDockLDockingPredictor(
        work_dir=str(tmp_path),
        runtime_dir=str(tmp_path / "runtime"),
        progress_heartbeat_s=0.01,
    )
    _set_required_inputs(
        module,
        BioSignal,
        protein_path="data/1a0q/1a0q_protein_processed.pdb",
        ligand_description="COc(cc1)ccc1C#N",
    )

    module.advance_to(0.5)
    captured = capsys.readouterr().out.splitlines()
    progress_events = [
        json.loads(line.removeprefix("BSIM_PROGRESS:"))
        for line in captured
        if line.startswith("BSIM_PROGRESS:")
    ]

    phases = {event["phase"] for event in progress_events}
    assert {"inputs", "runtime", "inference", "postprocess", "outputs", "completed"} <= phases
    assert any(
        event["phase"] == "inference" and "still running" in event["message"].lower()
        for event in progress_events
    )


def test_subprocess_failure_surfaces_metadata(biosim, tmp_path, monkeypatch):
    from src.diffdockl_docking_predictor import DiffDockLDockingPredictor
    from biosim.signals import BioSignal

    def fake_invoke(command, *, cwd, timeout, env, phase):  # noqa: ARG001
        if command[:4] == ["git", "clone", "--branch", "v1.1.3"]:
            repo_dir = Path(command[-1])
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / ".git").mkdir()
            (repo_dir / "default_inference_args.yaml").write_text("samples_per_complex: 10\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="cloned", stderr="")
        if "-m" in command and "venv" in command:
            venv_root = Path(command[-1])
            (venv_root / "bin").mkdir(parents=True, exist_ok=True)
            (venv_root / "bin" / "python").write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="venv", stderr="")
        if "-m" in command and "pip" in command and "install" in command:
            return subprocess.CompletedProcess(command, 0, stdout="pip", stderr="")
        return subprocess.CompletedProcess(command, 3, stdout="", stderr="boom")

    _patch_invoke_command(monkeypatch, DiffDockLDockingPredictor, fake_invoke)

    module = DiffDockLDockingPredictor(
        work_dir=str(tmp_path),
        runtime_dir=str(tmp_path / "runtime"),
    )
    _set_required_inputs(
        module,
        BioSignal,
        protein_path="data/1a0q/1a0q_protein_processed.pdb",
        ligand_description="COc(cc1)ccc1C#N",
    )
    module.advance_to(0.5)

    metadata = module.get_outputs()["run_metadata"].value
    assert metadata["status"] == "error"
    assert metadata["returncode"] == 3
    assert "non-zero" in metadata["error"]


def test_missing_expected_files_becomes_error(biosim, tmp_path, monkeypatch):
    from src.diffdockl_docking_predictor import DiffDockLDockingPredictor
    from biosim.signals import BioSignal

    def fake_invoke(command, *, cwd, timeout, env, phase):  # noqa: ARG001
        if command[:4] == ["git", "clone", "--branch", "v1.1.3"]:
            repo_dir = Path(command[-1])
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / ".git").mkdir()
            (repo_dir / "default_inference_args.yaml").write_text("samples_per_complex: 10\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="cloned", stderr="")
        if "-m" in command and "venv" in command:
            venv_root = Path(command[-1])
            (venv_root / "bin").mkdir(parents=True, exist_ok=True)
            (venv_root / "bin" / "python").write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="venv", stderr="")
        if "-m" in command and "pip" in command and "install" in command:
            return subprocess.CompletedProcess(command, 0, stdout="pip", stderr="")
        if command[1:3] == ["-m", "inference"]:
            out_dir = Path(command[command.index("--out_dir") + 1])
            complex_name = command[command.index("--complex_name") + 1]
            (out_dir / complex_name).mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    _patch_invoke_command(monkeypatch, DiffDockLDockingPredictor, fake_invoke)

    module = DiffDockLDockingPredictor(
        work_dir=str(tmp_path),
        runtime_dir=str(tmp_path / "runtime"),
    )
    _set_required_inputs(
        module,
        BioSignal,
        protein_path="data/1a0q/1a0q_protein_processed.pdb",
        ligand_description="COc(cc1)ccc1C#N",
    )
    module.advance_to(0.5)

    metadata = module.get_outputs()["run_metadata"].value
    assert metadata["status"] == "error"
    assert "expected DiffDock outputs" in metadata["error"]


def test_repeat_advance_does_not_rerun_until_reset(biosim, tmp_path, monkeypatch):
    from src.diffdockl_docking_predictor import DiffDockLDockingPredictor
    from biosim.signals import BioSignal

    calls = {"predict": 0}

    def fake_invoke(command, *, cwd, timeout, env, phase):  # noqa: ARG001
        if command[:4] == ["git", "clone", "--branch", "v1.1.3"]:
            repo_dir = Path(command[-1])
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / ".git").mkdir()
            (repo_dir / "default_inference_args.yaml").write_text("samples_per_complex: 10\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="cloned", stderr="")
        if "-m" in command and "venv" in command:
            venv_root = Path(command[-1])
            (venv_root / "bin").mkdir(parents=True, exist_ok=True)
            (venv_root / "bin" / "python").write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="venv", stderr="")
        if "-m" in command and "pip" in command and "install" in command:
            return subprocess.CompletedProcess(command, 0, stdout="pip", stderr="")
        if command[1:3] == ["-m", "inference"]:
            calls["predict"] += 1
            out_dir = Path(command[command.index("--out_dir") + 1])
            complex_name = command[command.index("--complex_name") + 1]
            prediction_dir = out_dir / complex_name
            prediction_dir.mkdir(parents=True, exist_ok=True)
            (prediction_dir / "rank1.sdf").write_text("rank1", encoding="utf-8")
            (prediction_dir / "rank1_confidence0.72.sdf").write_text("rank1-c", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    _patch_invoke_command(monkeypatch, DiffDockLDockingPredictor, fake_invoke)
    monkeypatch.setattr(
        DiffDockLDockingPredictor,
        "_ligand_pdb_lines_from_sdf",
        lambda self, _path, _python_executable=None: [
            "HETATM    1   C1 LIG Z   1      11.111  22.222  33.333  1.00  0.00           C"
        ],
    )

    module = DiffDockLDockingPredictor(
        work_dir=str(tmp_path),
        runtime_dir=str(tmp_path / "runtime"),
    )
    _set_required_inputs(
        module,
        BioSignal,
        protein_path="data/1a0q/1a0q_protein_processed.pdb",
        ligand_description="COc(cc1)ccc1C#N",
    )
    module.advance_to(0.2)
    module.advance_to(0.3)
    assert calls["predict"] == 1

    module.reset()
    _set_required_inputs(
        module,
        BioSignal,
        protein_path="data/1a0q/1a0q_protein_processed.pdb",
        ligand_description="COc(cc1)ccc1C#N",
    )
    module.advance_to(0.4)
    assert calls["predict"] == 2


def test_example_files_parse_and_reference_real_interface(biosim):
    repo_root = Path(__file__).resolve().parents[3]
    minimal = yaml.safe_load((repo_root / "examples" / "diffdock-minimal" / "config.yaml").read_text(encoding="utf-8"))
    wiring = yaml.safe_load((repo_root / "examples" / "diffdock-wiring" / "space.yaml").read_text(encoding="utf-8"))

    assert minimal["model"]["path"] == "../../models/diffdock-diffdockl-docking-predictor"
    assert minimal["model"]["inputs"]["protein_path"] == "data/1a0q/1a0q_protein_processed.pdb"
    assert minimal["model"]["inputs"]["ligand_description"] == "COc(cc1)ccc1C#N"
    assert minimal["model"]["inputs"]["run_options"]["samples_per_complex"] == 2
    assert wiring["models"][0]["path"] == "models/diffdock-diffdockl-docking-predictor"
    assert wiring["models"][0]["parameters"]["default_protein_path"] == "data/1a0q/1a0q_protein_processed.pdb"
    assert wiring["models"][0]["parameters"]["default_run_options"]["inference_steps"] == 4


def test_invoke_command_streams_stdout_and_stderr_and_preserves_raw_output(biosim, tmp_path, capsys):
    from src.diffdockl_docking_predictor import DiffDockLDockingPredictor

    module = DiffDockLDockingPredictor(work_dir=str(tmp_path))
    completed = module._invoke_command(
        [
            sys.executable,
            "-c",
            (
                "import sys, time; "
                "print('stdout-line', flush=True); "
                "sys.stderr.write('stderr-progress\\r'); sys.stderr.flush(); "
                "time.sleep(0.01); "
                "sys.stderr.write('stderr-line\\n'); sys.stderr.flush()"
            ),
        ],
        cwd=str(tmp_path),
        timeout=30.0,
        env=None,
        phase="runtime",
    )

    captured = capsys.readouterr()
    assert "[runtime] stdout-line" in captured.out
    assert "[runtime] stderr-progress" in captured.err
    assert "[runtime] stderr-line" in captured.err
    assert "stdout-line\n" in completed.stdout
    assert "stderr-progress\rstderr-line\n" in completed.stderr


def test_ligand_pdb_lines_fall_back_to_runtime_python_when_rdkit_is_missing(biosim, tmp_path, monkeypatch):
    from src.diffdockl_docking_predictor import DiffDockLDockingPredictor

    real_run = subprocess.run
    calls: list[list[str]] = []

    def fake_run(command, cwd=None, capture_output=False, text=False, timeout=None, check=False, env=None):  # noqa: ARG001
        command = [str(item) for item in command]
        if command[1:3] == ["-c", command[2]]:
            calls.append(command)
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    [
                        "HETATM    1   C1 LIG Z   1      11.111  22.222  33.333  1.00  0.00           C"
                    ]
                ),
                stderr="",
            )
        return real_run(
            command,
            cwd=cwd,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            check=check,
            env=env,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        DiffDockLDockingPredictor,
        "_ligand_pdb_lines_from_sdf_inline",
        lambda self, _path: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'rdkit'")),
    )

    module = DiffDockLDockingPredictor(work_dir=str(tmp_path))
    sdf_path = tmp_path / "rank1.sdf"
    sdf_path.write_text("dummy", encoding="utf-8")

    lines = module._ligand_pdb_lines_from_sdf(sdf_path, "/tmp/runtime-python")

    assert lines == [
        "HETATM    1   C1 LIG Z   1      11.111  22.222  33.333  1.00  0.00           C"
    ]
    assert calls
    assert calls[0][0] == "/tmp/runtime-python"


@pytest.mark.skipif(
    os.getenv("BIOSIM_DIFFDOCK_RUN_REAL_SMOKE") != "1",
    reason="Set BIOSIM_DIFFDOCK_RUN_REAL_SMOKE=1 to run the real DiffDock smoke test.",
)
def test_real_smoke_example_runs(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    output_json = tmp_path / "real-smoke-output.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "run_example.py"),
            "diffdock-minimal",
            "--work-dir",
            str(tmp_path / "runs"),
            "--runtime-dir",
            str(tmp_path / "runtime"),
            "--output-json",
            str(output_json),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=14_400,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    outputs = payload["outputs"]
    assert outputs["run_metadata"]["value"]["status"] == "completed"
    assert Path(outputs["structure_artifacts"]["value"]["top_complex_file"]).exists()
    assert Path(outputs["structure_artifacts"]["value"]["top_pose_file"]).exists()
