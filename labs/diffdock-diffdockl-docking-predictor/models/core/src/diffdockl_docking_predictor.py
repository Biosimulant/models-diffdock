# SPDX-FileCopyrightText: 2026-present Biosimulant Team
#
# SPDX-License-Identifier: MIT
"""DiffDock-L single-complex docking wrapper."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from biosim import BioModule
from biosim.signals import (AcceptedSignalProfile, ArraySignal, BioSignal, EventSignal, RecordSignal, ScalarSignal, SignalSpec)


_ALLOWED_RUN_OPTIONS = {
    "complex_name",
    "samples_per_complex",
    "inference_steps",
    "batch_size",
    "save_visualisation",
}
_PIP_OPTION_FLAGS = {"--extra-index-url", "--find-links"}
_PREINSTALL_PACKAGE_NAMES = {"setuptools", "torch"}
_PATH_LIKE_SUFFIXES = {".pdb", ".sdf", ".mol2", ".mol", ".pdbqt"}
_POSE_FILE_RE = re.compile(
    r"^rank(?P<rank>\d+)(?:_confidence(?P<confidence>[-+]?\d+(?:\.\d+)?))?\.sdf$"
)
_RDKIT_SDF_TO_PDB_JSON_SCRIPT = textwrap.dedent(
    """
    import json
    import sys
    from rdkit import Chem

    sdf_path = sys.argv[1]
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    molecule = next((mol for mol in supplier if mol is not None), None)
    if molecule is None:
        raise RuntimeError(f"could not read ligand pose from {sdf_path}")
    pdb_block = Chem.MolToPDBBlock(molecule)

    lines = []
    serial = 1
    for raw in pdb_block.splitlines():
        if not raw.startswith(("ATOM", "HETATM")):
            continue
        atom_name = raw[12:16].strip() or "C"
        element = raw[76:78].strip() or atom_name[:2].strip().upper()
        x = float(raw[30:38])
        y = float(raw[38:46])
        z = float(raw[46:54])
        lines.append(
            f"HETATM{serial:5d} {atom_name:>4} LIG Z{1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2}"
        )
        serial += 1
    if not lines:
        raise RuntimeError(f"no ligand atoms were extracted from {sdf_path}")

    print(json.dumps(lines))
    """
).strip()


def _split_completed_output_fragments(text: str) -> tuple[list[str], str]:
    if not text:
        return [], ""

    completed: list[str] = []
    start = 0
    index = 0
    text_length = len(text)
    while index < text_length:
        char = text[index]
        if char not in "\r\n":
            index += 1
            continue
        terminator_len = 1
        if char == "\r" and index + 1 < text_length and text[index + 1] == "\n":
            terminator_len = 2
        fragment = text[start:index]
        if fragment.strip():
            completed.append(fragment)
        start = index + terminator_len
        index = start
    return completed, text[start:]


def _coerce_string(value: Any, *preferred_keys: str) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Mapping):
        for key in preferred_keys:
            candidate = value.get(key)
            if isinstance(candidate, str):
                text = candidate.strip()
                if text:
                    return text
    return None


def _coerce_run_options(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(key, str):
            out[key] = item
    return out


def _schema_type(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return "json"


def _signal_value(signal):
    value = signal.value
    if isinstance(value, dict) and set(value.keys()) == {"payload"}:
        return value["payload"]
    return value


def _generic_input_spec(description=None):
    return SignalSpec.record(
        schema={"payload": "json"},
        accepted_profiles=(
            AcceptedSignalProfile(signal_type="record", schema={"payload": "json"}),
            AcceptedSignalProfile(signal_type="scalar"),
        ),
        description=description,
    )


def _make_signal(*, source, name, value, emitted_at, spec=None):
    if spec is None:
        if isinstance(value, dict):
            spec = SignalSpec.record(schema={str(key): _schema_type(item) for key, item in value.items()})
        elif isinstance(value, (list, tuple)):
            spec = SignalSpec.record(schema={"payload": "json"})
        else:
            spec = SignalSpec.scalar(dtype=_schema_type(value))

    if spec.signal_type == "scalar":
        return ScalarSignal(source=source, name=name, value=value, emitted_at=emitted_at, spec=spec)
    if spec.signal_type == "array":
        return ArraySignal(source=source, name=name, value=value, emitted_at=emitted_at, spec=spec)
    if spec.signal_type == "event":
        event_value = value
        if spec.schema is not None and not (isinstance(value, dict) and set(value.keys()) == set(spec.schema.keys())):
            event_value = {"payload": value}
        return EventSignal(source=source, name=name, value=event_value, emitted_at=emitted_at, spec=spec)

    record_value = value
    if not isinstance(value, dict) or set(value.keys()) != set((spec.schema or {}).keys()):
        record_value = {"payload": value}
    return RecordSignal(source=source, name=name, value=record_value, emitted_at=emitted_at, spec=spec)

class DiffDockLDockingPredictor(BioModule):
    """Run upstream DiffDock-L for a single receptor PDB plus ligand input."""

    def __init__(
        self,
        default_protein_path: Optional[str] = None,
        default_ligand_description: Optional[str] = None,
        default_run_options: Optional[Mapping[str, Any]] = None,
        runtime_mode: str = "managed",
        runtime_dir: Optional[str] = None,
        runtime_python: Optional[str] = None,
        requirements_file: Optional[str] = None,
        diffdock_repo_url: str = "https://github.com/gcorso/DiffDock.git",
        diffdock_git_ref: str = "v1.1.3",
        work_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        command_timeout_s: float = 10_800.0,
        runtime_setup_timeout_s: float = 7_200.0,
        progress_heartbeat_s: float = 30.0,
        integration_step: float = 0.01,
    ) -> None:
        self.integration_step = float(integration_step)
        self.runtime_mode = runtime_mode
        self.runtime_python = runtime_python
        self.diffdock_repo_url = diffdock_repo_url
        self.diffdock_git_ref = diffdock_git_ref
        self.command_timeout_s = command_timeout_s
        self.runtime_setup_timeout_s = runtime_setup_timeout_s
        self.progress_heartbeat_s = max(0.0, float(progress_heartbeat_s))
        self.work_dir = Path(work_dir).expanduser().resolve() if work_dir else None

        self.model_root = Path(__file__).resolve().parents[1]
        repo_root = self.model_root.parents[1]
        self.runtime_dir = (
            Path(runtime_dir).expanduser().resolve()
            if runtime_dir
            else (repo_root / ".runtime" / "diffdock").resolve()
        )
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir
            else (self.runtime_dir / "cache").resolve()
        )
        self.requirements_file = (
            Path(requirements_file).expanduser().resolve()
            if requirements_file
            else (self.model_root / "requirements" / "runtime-gpu.txt").resolve()
        )

        self._protein_path: Optional[str] = _coerce_string(default_protein_path, "path")
        self._ligand_description: Optional[str] = _coerce_string(
            default_ligand_description, "path", "smiles", "description"
        )
        self._run_options: dict[str, Any] = _coerce_run_options(default_run_options)
        self._outputs: dict[str, BioSignal] = {}
        self._cached_payloads: dict[str, Any] = {}
        self._last_signature: Optional[str] = None

    def inputs(self) -> dict[str, SignalSpec]:
        return {
            'protein_path': _generic_input_spec(),
            'ligand_description': _generic_input_spec(),
            'run_options': _generic_input_spec(),
        }

    def outputs(self) -> dict[str, SignalSpec]:
        return {
            'pose_summary': SignalSpec.record(schema={'payload': 'json'}),
            'confidence_summary': SignalSpec.record(schema={'payload': 'json'}),
            'structure_artifacts': SignalSpec.record(schema={'payload': 'json'}),
            'run_metadata': SignalSpec.record(schema={'payload': 'json'}),
        }

    def reset(self) -> None:
        self._outputs = {}
        self._cached_payloads = {}
        self._last_signature = None

    def set_inputs(self, signals: dict[str, BioSignal]) -> None:
        changed = False

        protein_signal = signals.get("protein_path")
        if protein_signal is not None:
            protein_path = _coerce_string(_signal_value(protein_signal), "path")
            if protein_path != self._protein_path:
                self._protein_path = protein_path
                changed = True

        ligand_signal = signals.get("ligand_description")
        if ligand_signal is not None:
            ligand_description = _coerce_string(
                _signal_value(ligand_signal), "path", "smiles", "description"
            )
            if ligand_description != self._ligand_description:
                self._ligand_description = ligand_description
                changed = True

        run_signal = signals.get("run_options")
        if run_signal is not None:
            run_options = _coerce_run_options(_signal_value(run_signal))
            if run_options != self._run_options:
                self._run_options = run_options
                changed = True

        if changed:
            self._last_signature = None

    def advance_window(self, start: float, end: float) -> None:
        t = float(end)
        metadata: dict[str, Any] = {
            "status": "running",
            "runtime_mode": self.runtime_mode,
            "runtime_dir": str(self.runtime_dir),
            "cache_dir": str(self.cache_dir),
            "requirements_file": str(self.requirements_file),
            "repo_bootstrapped": False,
            "runtime_bootstrapped": False,
            "runtime_setup_commands": [],
            "stdout": "",
            "stderr": "",
        }

        self._emit_progress("inputs", "Validating DiffDock inputs")
        try:
            resolved_options = self._resolved_options()
            protein_path = self._resolve_required_input_path(
                self._protein_path,
                input_name="protein_path",
            )
            ligand_input = self._resolve_ligand_description(self._ligand_description)
        except Exception as exc:  # noqa: BLE001
            metadata["status"] = "error"
            metadata["error"] = str(exc)
            self._emit_progress("error", metadata["error"])
            self._set_error_payload(str(exc), metadata=metadata)
            self._emit_outputs(t)
            return

        signature = json.dumps(
            {
                "protein_path": protein_path,
                "ligand": ligand_input,
                "run_options": resolved_options,
            },
            sort_keys=True,
        )
        if signature == self._last_signature and self._cached_payloads:
            self._emit_progress("cache", "Reusing cached DiffDock outputs for unchanged inputs")
            self._emit_outputs(t)
            return

        run_root = self._create_run_root()
        output_dir = run_root / "output"
        metadata["run_root"] = str(run_root)
        metadata["output_dir"] = str(output_dir)
        metadata["protein_path"] = protein_path
        metadata["ligand_description"] = ligand_input["value"]

        try:
            self._emit_progress("runtime", "Preparing DiffDock runtime")
            runtime = self._prepare_runtime(run_root, metadata)
            command = self._build_command(
                python_executable=runtime["python_executable"],
                repo_dir=runtime["repo_dir"],
                output_dir=output_dir,
                protein_path=protein_path,
                ligand_input=ligand_input,
                options=resolved_options,
            )
            metadata["resolved_python_executable"] = runtime["python_executable"]
            metadata["repo_dir"] = str(runtime["repo_dir"])
            metadata["model_dir"] = str(runtime["model_dir"])
            metadata["confidence_model_dir"] = str(runtime["confidence_model_dir"])
            metadata["command"] = command

            completed = self._run_command_with_progress(
                command=command,
                cwd=runtime["repo_dir"],
                timeout=self.command_timeout_s,
                env=self._command_env(runtime["repo_dir"]),
                phase="inference",
                start_message=f"Starting DiffDock-L inference for {resolved_options['complex_name']}",
                heartbeat_message="DiffDock-L inference is still running",
                completion_message="DiffDock-L inference finished",
            )
            metadata["returncode"] = completed.returncode
            metadata["stdout"] = (completed.stdout or "")[-12000:]
            metadata["stderr"] = (completed.stderr or "")[-12000:]
        except Exception as exc:  # noqa: BLE001
            metadata["status"] = "error"
            metadata["error"] = f"failed to execute DiffDock: {exc}"
            self._emit_progress("error", metadata["error"])
            self._set_error_payload(metadata["error"], metadata=metadata)
            self._last_signature = signature
            self._emit_outputs(t)
            return

        if completed.returncode != 0:
            metadata["status"] = "error"
            metadata["error"] = "DiffDock inference returned a non-zero exit code"
            self._emit_progress("error", metadata["error"])
            self._set_error_payload(metadata["error"], metadata=metadata)
            self._last_signature = signature
            self._emit_outputs(t)
            return

        try:
            self._emit_progress("postprocess", "Collecting ranked poses and confidence scores")
            prediction_dir = self._find_prediction_dir(output_dir, resolved_options["complex_name"])
            pose_records, top_pose_path, reverseprocess_files = self._collect_pose_records(prediction_dir)
            confidence_summary = self._build_confidence_summary(pose_records)
            pose_summary_file = prediction_dir / "pose_summary.json"
            confidence_file = prediction_dir / "confidence_summary.json"
            self._write_json(pose_summary_file, {"poses": pose_records})
            self._write_json(confidence_file, confidence_summary)
            top_complex_file = prediction_dir / "top_rank_complex.pdb"
            self._emit_progress("postprocess", "Building merged protein-ligand complex")
            self._build_top_rank_complex(
                protein_path=Path(protein_path),
                ligand_pose_path=top_pose_path,
                output_path=top_complex_file,
                python_executable=runtime["python_executable"],
            )
            self._emit_progress("outputs", "Publishing DiffDock artifacts and visuals")
            artifacts = self._build_structure_artifacts(
                prediction_dir=prediction_dir,
                pose_records=pose_records,
                top_pose_path=top_pose_path,
                top_complex_file=top_complex_file,
                confidence_file=confidence_file,
                pose_summary_file=pose_summary_file,
                reverseprocess_files=reverseprocess_files,
            )
        except Exception as exc:  # noqa: BLE001
            metadata["status"] = "error"
            metadata["error"] = f"expected DiffDock outputs were not found: {exc}"
            self._emit_progress("error", metadata["error"])
            self._set_error_payload(metadata["error"], metadata=metadata)
            self._last_signature = signature
            self._emit_outputs(t)
            return

        metadata["status"] = "completed"
        metadata["prediction_dir"] = str(prediction_dir)
        self._cached_payloads = {
            "pose_summary": pose_records,
            "confidence_summary": confidence_summary,
            "structure_artifacts": artifacts,
            "run_metadata": metadata,
        }
        self._last_signature = signature
        self._emit_progress("completed", "DiffDock outputs are ready")
        self._emit_outputs(t)

    def get_outputs(self) -> dict[str, BioSignal]:
        return dict(self._outputs)

    def visualize(self) -> Optional[list[dict[str, Any]]]:
        return None

    def _resolved_options(self) -> dict[str, Any]:
        resolved: dict[str, Any] = {
            "complex_name": "complex_0",
            "samples_per_complex": 10,
            "inference_steps": 20,
            "batch_size": 10,
            "save_visualisation": False,
        }
        for key in self._run_options:
            if key not in _ALLOWED_RUN_OPTIONS:
                raise ValueError(f"unsupported run_options key: {key}")

        if "complex_name" in self._run_options:
            complex_name = _coerce_string(self._run_options.get("complex_name"))
            if complex_name is None:
                raise ValueError("run_options.complex_name must be a non-empty string")
            resolved["complex_name"] = complex_name

        for key in ("samples_per_complex", "inference_steps", "batch_size"):
            if key in self._run_options:
                value = self._run_options.get(key)
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"run_options.{key} must be a positive integer")
                resolved[key] = value

        if "save_visualisation" in self._run_options:
            value = self._run_options.get("save_visualisation")
            if not isinstance(value, bool):
                raise ValueError("run_options.save_visualisation must be a boolean")
            resolved["save_visualisation"] = value

        return resolved

    def _create_run_root(self) -> Path:
        base_dir = self.work_dir
        if base_dir is not None:
            base_dir.mkdir(parents=True, exist_ok=True)
        root = Path(
            tempfile.mkdtemp(
                prefix="diffdock-run-",
                dir=str(base_dir) if base_dir is not None else None,
            )
        )
        return root.resolve()

    def _prepare_runtime(self, run_root: Path, metadata: dict[str, Any]) -> dict[str, Path | str]:
        runtime_root = self.runtime_dir
        runtime_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        repo_dir = runtime_root / "repo"
        self._ensure_repo_checkout(repo_dir, run_root, metadata)

        if self.runtime_mode.strip().lower() == "managed":
            venv_dir = runtime_root / "venv"
            python_executable = self._ensure_managed_runtime(repo_dir, venv_dir, run_root, metadata)
        elif self.runtime_mode.strip().lower() == "external":
            python_executable = self._resolve_external_python()
            self._emit_progress(
                "runtime",
                f"Using external DiffDock runtime at {python_executable}",
            )
        else:
            raise ValueError(f"unsupported runtime_mode: {self.runtime_mode}")

        return {
            "repo_dir": repo_dir,
            "python_executable": python_executable,
            "model_dir": repo_dir / "workdir" / "v1.1" / "score_model",
            "confidence_model_dir": repo_dir / "workdir" / "v1.1" / "confidence_model",
        }

    def _ensure_repo_checkout(self, repo_dir: Path, run_root: Path, metadata: dict[str, Any]) -> None:
        if (repo_dir / ".git").is_dir():
            self._emit_progress("runtime", "Reusing cached DiffDock repository checkout")
            return
        if repo_dir.exists():
            raise RuntimeError(f"runtime repo path exists but is not a git checkout: {repo_dir}")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        self._run_setup_command(
            [
                "git",
                "clone",
                "--branch",
                self.diffdock_git_ref,
                "--depth",
                "1",
                self.diffdock_repo_url,
                str(repo_dir),
            ],
            run_root,
            metadata,
            phase="runtime",
            start_message=f"Cloning DiffDock repository at {self.diffdock_git_ref}",
            heartbeat_message="Still cloning the DiffDock repository",
            completion_message="DiffDock repository checkout is ready",
        )
        metadata["repo_bootstrapped"] = True

    def _ensure_managed_runtime(
        self,
        repo_dir: Path,
        venv_dir: Path,
        run_root: Path,
        metadata: dict[str, Any],
    ) -> str:
        python_executable = venv_dir / "bin" / "python"
        ready_marker = venv_dir / ".ready"
        requirements_hash = hashlib.sha256(
            (self.requirements_file.read_text(encoding="utf-8") + self.diffdock_git_ref).encode("utf-8")
        ).hexdigest()[:16]

        if not python_executable.exists():
            if venv_dir.exists():
                shutil.rmtree(venv_dir, ignore_errors=True)
            self._run_setup_command(
                [sys.executable, "-m", "venv", str(venv_dir)],
                run_root,
                metadata,
                phase="runtime",
                start_message="Creating managed DiffDock runtime environment",
                heartbeat_message="Still creating the managed runtime environment",
                completion_message="Managed DiffDock runtime environment is ready",
            )
            metadata["runtime_bootstrapped"] = True

        if not ready_marker.is_file() or ready_marker.read_text(encoding="utf-8").strip() != requirements_hash:
            self._run_setup_command(
                [str(python_executable), "-m", "pip", "install", "--upgrade", "pip"],
                run_root,
                metadata,
                phase="runtime",
                start_message="Upgrading pip inside the managed DiffDock runtime",
                heartbeat_message="Still upgrading pip inside the managed runtime",
                completion_message="Managed runtime pip is up to date",
            )
            option_args, package_specs = self._parse_requirements_install_args()
            install_phases = self._pip_install_phases(package_specs)
            for index, package_phase in enumerate(install_phases, start=1):
                self._run_setup_command(
                    [
                        str(python_executable),
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "--no-input",
                        *option_args,
                        *package_phase,
                    ],
                    run_root,
                    metadata,
                    phase="runtime",
                    start_message=(
                        f"Installing DiffDock runtime dependencies "
                        f"(phase {index} of {len(install_phases)})"
                    ),
                    heartbeat_message=(
                        f"Still installing DiffDock runtime dependencies "
                        f"(phase {index} of {len(install_phases)})"
                    ),
                    completion_message=(
                        f"Finished installing DiffDock runtime dependencies "
                        f"(phase {index} of {len(install_phases)})"
                    ),
                )
            ready_marker.write_text(requirements_hash, encoding="utf-8")
            metadata["runtime_bootstrapped"] = True
        else:
            self._emit_progress("runtime", "Reusing cached managed DiffDock runtime")

        return str(python_executable)

    def _run_setup_command(
        self,
        command: list[str],
        run_root: Path,
        metadata: dict[str, Any],
        *,
        phase: str = "runtime",
        start_message: Optional[str] = None,
        heartbeat_message: Optional[str] = None,
        completion_message: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        completed = self._run_command_with_progress(
            command=command,
            cwd=run_root,
            timeout=self.runtime_setup_timeout_s,
            env=None,
            phase=phase,
            start_message=start_message or f"Running setup command: {' '.join(command[:3])}",
            heartbeat_message=heartbeat_message or "Still running setup command",
            completion_message=completion_message or "Setup command finished",
        )
        metadata["runtime_setup_commands"].append(
            {
                "command": list(command),
                "cwd": str(run_root),
                "returncode": completed.returncode,
                "stdout": (completed.stdout or "")[-8000:],
                "stderr": (completed.stderr or "")[-8000:],
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"setup command failed ({completed.returncode}): {' '.join(command)}"
            )
        return completed

    def _run_command_with_progress(
        self,
        *,
        command: list[str],
        cwd: Path,
        timeout: float,
        env: Optional[dict[str, str]],
        phase: str,
        start_message: str,
        heartbeat_message: str,
        completion_message: str,
    ) -> subprocess.CompletedProcess[str]:
        self._emit_progress(phase, start_message)

        outcome: dict[str, subprocess.CompletedProcess[str]] = {}
        error: dict[str, BaseException] = {}

        def runner() -> None:
            try:
                outcome["completed"] = self._invoke_command(
                    command,
                    cwd=str(cwd),
                    timeout=timeout,
                    env=env,
                    phase=phase,
                )
            except BaseException as exc:  # noqa: BLE001
                error["exc"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()

        started_at = time.monotonic()
        tick = 0
        while thread.is_alive():
            join_timeout = self.progress_heartbeat_s or None
            thread.join(timeout=join_timeout)
            if not thread.is_alive():
                break
            tick += 1
            elapsed = int(time.monotonic() - started_at)
            self._emit_progress(
                phase,
                f"{heartbeat_message} ({elapsed}s elapsed)",
                tick=tick,
                duration=float(elapsed),
            )

        thread.join()
        if "exc" in error:
            raise error["exc"]

        completed = outcome["completed"]
        if completed.returncode == 0:
            self._emit_progress(phase, completion_message)
        else:
            self._emit_progress(
                phase,
                f"{completion_message} (exit code {completed.returncode})",
            )
        return completed

    def _invoke_command(
        self,
        command: list[str],
        *,
        cwd: str,
        timeout: float,
        env: Optional[dict[str, str]],
        phase: str,
    ) -> subprocess.CompletedProcess[str]:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
            env=env,
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("failed to open subprocess stdout/stderr pipes")

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stdout_remainder = [""]
        stderr_remainder = [""]

        stdout_thread = threading.Thread(
            target=self._pump_stream,
            args=(process.stdout, stdout_chunks, stdout_remainder, sys.stdout, phase),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._pump_stream,
            args=(process.stderr, stderr_chunks, stderr_remainder, sys.stderr, phase),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        finally:
            stdout_thread.join()
            stderr_thread.join()

        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )

    def _pump_stream(
        self,
        handle: Any,
        chunks: list[str],
        remainder: list[str],
        target: Any,
        phase: str,
    ) -> None:
        while True:
            chunk = handle.read(1)
            if chunk in (b"", ""):
                break
            text = chunk.decode(errors="ignore") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
            chunks.append(text)
            remainder[0] += text
            completed, tail = _split_completed_output_fragments(remainder[0])
            remainder[0] = tail
            for message in completed:
                self._emit_forwarded_command_log(phase, message, target)
        if remainder[0].strip():
            self._emit_forwarded_command_log(phase, remainder[0], target)
            remainder[0] = ""

    def _emit_forwarded_command_log(self, phase: str, message: str, target: Any) -> None:
        target.write(f"[{phase}] {message}\n")
        target.flush()

    def _parse_requirements_install_args(self) -> tuple[list[str], list[str]]:
        option_args: list[str] = []
        package_specs: list[str] = []
        for raw in self.requirements_file.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("-"):
                parts = stripped.split(None, 1)
                option = parts[0]
                if option not in _PIP_OPTION_FLAGS or len(parts) != 2:
                    raise RuntimeError(f"unsupported requirements option: {stripped}")
                option_args.extend([option, parts[1].strip()])
                continue
            package_specs.append(stripped)
        return option_args, package_specs

    def _pip_install_phases(self, package_specs: list[str]) -> list[list[str]]:
        preinstall: list[str] = []
        remainder: list[str] = []
        for spec in package_specs:
            name = spec.split("==", 1)[0].strip().lower()
            if name in _PREINSTALL_PACKAGE_NAMES:
                preinstall.append(spec)
            else:
                remainder.append(spec)
        phases: list[list[str]] = []
        if preinstall:
            phases.append(preinstall)
        if remainder:
            phases.append(remainder)
        return phases

    def _resolve_external_python(self) -> str:
        if isinstance(self.runtime_python, str) and self.runtime_python.strip():
            candidate = Path(self.runtime_python).expanduser()
            if candidate.is_file():
                return str(candidate.resolve())
        for name in ("python", "python3"):
            resolved = shutil.which(name)
            if resolved:
                return resolved
        return sys.executable

    def _command_env(self, repo_dir: Path) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("TORCH_HOME", str((self.cache_dir / "torch").resolve()))
        env.setdefault("HF_HOME", str((self.cache_dir / "huggingface").resolve()))
        env.setdefault("XDG_CACHE_HOME", str((self.cache_dir / "xdg").resolve()))
        env.setdefault("REPOSITORY_URL", self.diffdock_repo_url.removesuffix(".git"))
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONNOUSERSITE", "1")
        env.setdefault("DIFFDOCK_REPO_ROOT", str(repo_dir))
        return env

    def _build_command(
        self,
        *,
        python_executable: str,
        repo_dir: Path,
        output_dir: Path,
        protein_path: str,
        ligand_input: dict[str, str],
        options: dict[str, Any],
    ) -> list[str]:
        command = [
            python_executable,
            "-m",
            "inference",
            "--config",
            str((repo_dir / "default_inference_args.yaml").resolve()),
            "--out_dir",
            str(output_dir.resolve()),
            "--complex_name",
            options["complex_name"],
            "--protein_path",
            protein_path,
            "--ligand_description",
            ligand_input["value"],
            "--model_dir",
            str((repo_dir / "workdir" / "v1.1" / "score_model").resolve()),
            "--confidence_model_dir",
            str((repo_dir / "workdir" / "v1.1" / "confidence_model").resolve()),
            "--ckpt",
            "best_ema_inference_epoch_model.pt",
            "--confidence_ckpt",
            "best_model_epoch75.pt",
            "--samples_per_complex",
            str(options["samples_per_complex"]),
            "--inference_steps",
            str(options["inference_steps"]),
            "--batch_size",
            str(options["batch_size"]),
        ]
        if options["save_visualisation"]:
            command.append("--save_visualisation")
        return command

    def _resolve_required_input_path(self, raw: Optional[str], *, input_name: str) -> str:
        if raw is None:
            raise ValueError(f"{input_name} input is required")
        for candidate in self._path_candidates(raw):
            if candidate.exists() and candidate.is_file():
                return str(candidate.resolve())
        raise FileNotFoundError(f"{input_name} file does not exist: {raw}")

    def _resolve_ligand_description(self, raw: Optional[str]) -> dict[str, str]:
        if raw is None:
            raise ValueError("ligand_description input is required")
        if self._looks_like_path(raw):
            for candidate in self._path_candidates(raw):
                if candidate.exists() and candidate.is_file():
                    return {"kind": "path", "value": str(candidate.resolve())}
            raise FileNotFoundError(f"ligand_description file does not exist: {raw}")
        return {"kind": "smiles", "value": raw}

    def _path_candidates(self, raw: str) -> list[Path]:
        path = Path(raw).expanduser()
        if path.is_absolute():
            return [path]
        return [(self.model_root / path).resolve(), (Path.cwd() / path).resolve()]

    def _looks_like_path(self, value: str) -> bool:
        if value.startswith(".") or "/" in value or "\\" in value:
            return True
        return Path(value).suffix.lower() in _PATH_LIKE_SUFFIXES

    def _find_prediction_dir(self, output_dir: Path, complex_name: str) -> Path:
        direct = (output_dir / complex_name).resolve()
        if direct.is_dir():
            return direct
        candidates = sorted(path for path in output_dir.rglob("rank1.sdf") if path.is_file())
        if not candidates:
            candidates = sorted(path for path in output_dir.rglob("rank1_confidence*.sdf") if path.is_file())
        if not candidates:
            raise FileNotFoundError(f"no DiffDock rank outputs found under {output_dir}")
        return candidates[0].parent.resolve()

    def _collect_pose_records(
        self,
        prediction_dir: Path,
    ) -> tuple[list[dict[str, Any]], Path, dict[int, Path]]:
        stable_pose_files: dict[int, Path] = {}
        ranked_pose_files: dict[int, tuple[Path, Optional[float]]] = {}
        reverseprocess_files: dict[int, Path] = {}

        for path in sorted(prediction_dir.glob("rank*.sdf")):
            match = _POSE_FILE_RE.match(path.name)
            if match is None:
                continue
            rank = int(match.group("rank"))
            confidence_raw = match.group("confidence")
            if confidence_raw is None:
                stable_pose_files[rank] = path.resolve()
                continue
            ranked_pose_files[rank] = (path.resolve(), float(confidence_raw))

        for path in sorted(prediction_dir.glob("rank*_reverseprocess.pdb")):
            name = path.stem.split("_", 1)[0]
            if not name.startswith("rank"):
                continue
            try:
                rank = int(name.replace("rank", "", 1))
            except ValueError:
                continue
            reverseprocess_files[rank] = path.resolve()

        ranks = sorted(set(stable_pose_files) | set(ranked_pose_files))
        if not ranks:
            raise FileNotFoundError(f"no DiffDock pose files found under {prediction_dir}")

        pose_records: list[dict[str, Any]] = []
        top_pose_path: Optional[Path] = None
        for rank in ranks:
            ranked_path, confidence = ranked_pose_files.get(rank, (None, None))
            stable_path = stable_pose_files.get(rank)
            file_path = ranked_path or stable_path
            if file_path is None:
                continue
            if rank == 1:
                top_pose_path = stable_path or ranked_path
            pose_records.append(
                {
                    "rank": rank,
                    "confidence": confidence,
                    "confidence_band": self._confidence_band(confidence),
                    "file_path": str(file_path),
                }
            )

        if top_pose_path is None:
            top_pose_path = Path(pose_records[0]["file_path"]).resolve()
        return pose_records, top_pose_path.resolve(), reverseprocess_files

    def _build_confidence_summary(self, pose_records: list[dict[str, Any]]) -> dict[str, Any]:
        top = pose_records[0]
        confidences = [
            float(item["confidence"])
            for item in pose_records
            if isinstance(item.get("confidence"), (int, float))
        ]
        return {
            "top_pose_rank": top["rank"],
            "top_pose_confidence": top.get("confidence"),
            "confidence_band": top.get("confidence_band"),
            "pose_count": len(pose_records),
            "all_confidences": confidences,
        }

    def _build_structure_artifacts(
        self,
        *,
        prediction_dir: Path,
        pose_records: list[dict[str, Any]],
        top_pose_path: Path,
        top_complex_file: Path,
        confidence_file: Path,
        pose_summary_file: Path,
        reverseprocess_files: dict[int, Path],
    ) -> dict[str, Any]:
        artifacts: dict[str, Any] = {
            "prediction_dir": str(prediction_dir),
            "top_pose_file": str(top_pose_path),
            "top_complex_file": str(top_complex_file),
            "confidence_file": str(confidence_file),
            "pose_summary_file": str(pose_summary_file),
        }
        for pose in pose_records:
            rank = int(pose["rank"])
            artifacts[f"rank_{rank}_file"] = str(Path(str(pose["file_path"])))
        for rank, path in reverseprocess_files.items():
            artifacts[f"rank_{rank}_reverseprocess_file"] = str(path)
        return artifacts

    def _build_top_rank_complex(
        self,
        *,
        protein_path: Path,
        ligand_pose_path: Path,
        output_path: Path,
        python_executable: Optional[str] = None,
    ) -> None:
        protein_lines = []
        for raw in protein_path.read_text(encoding="utf-8").splitlines():
            if raw.startswith("END"):
                continue
            protein_lines.append(raw)
        ligand_lines = self._ligand_pdb_lines_from_sdf(ligand_pose_path, python_executable)
        output_path.write_text(
            "\n".join(protein_lines + ["TER"] + ligand_lines + ["END"]) + "\n",
            encoding="utf-8",
        )

    def _ligand_pdb_lines_from_sdf(
        self,
        sdf_path: Path,
        python_executable: Optional[str] = None,
    ) -> list[str]:
        try:
            return self._ligand_pdb_lines_from_sdf_inline(sdf_path)
        except ModuleNotFoundError as exc:
            if not python_executable:
                raise
            return self._ligand_pdb_lines_from_sdf_via_runtime(
                sdf_path,
                python_executable=python_executable,
                original_error=exc,
            )

    def _ligand_pdb_lines_from_sdf_inline(self, sdf_path: Path) -> list[str]:
        from rdkit import Chem

        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        molecule = next((mol for mol in supplier if mol is not None), None)
        if molecule is None:
            raise RuntimeError(f"could not read ligand pose from {sdf_path}")
        pdb_block = Chem.MolToPDBBlock(molecule)

        lines: list[str] = []
        serial = 1
        for raw in pdb_block.splitlines():
            if not raw.startswith(("ATOM", "HETATM")):
                continue
            atom_name = raw[12:16].strip() or "C"
            element = raw[76:78].strip() or atom_name[:2].strip().upper()
            x = float(raw[30:38])
            y = float(raw[38:46])
            z = float(raw[46:54])
            lines.append(
                f"HETATM{serial:5d} {atom_name:>4} LIG Z{1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2}"
            )
            serial += 1
        if not lines:
            raise RuntimeError(f"no ligand atoms were extracted from {sdf_path}")
        return lines

    def _ligand_pdb_lines_from_sdf_via_runtime(
        self,
        sdf_path: Path,
        *,
        python_executable: str,
        original_error: ModuleNotFoundError,
    ) -> list[str]:
        completed = subprocess.run(
            [python_executable, "-c", _RDKIT_SDF_TO_PDB_JSON_SCRIPT, str(sdf_path)],
            capture_output=True,
            text=True,
            timeout=min(self.runtime_setup_timeout_s, 600.0),
            check=False,
            cwd=str(sdf_path.parent),
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or str(original_error)
            raise RuntimeError(
                f"could not convert ligand pose with runtime python {python_executable}: {detail}"
            ) from original_error
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"runtime python {python_executable} returned invalid ligand conversion output"
            ) from exc
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise RuntimeError(
                f"runtime python {python_executable} returned an unexpected ligand conversion payload"
            )
        if not payload:
            raise RuntimeError(f"no ligand atoms were extracted from {sdf_path}")
        return payload

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _emit_progress(self, phase: str, message: str, **payload: Any) -> None:
        event: dict[str, Any] = {"phase": phase, "message": message}
        for key, value in payload.items():
            if value is not None:
                event[key] = value
        print(f"BSIM_PROGRESS:{json.dumps(event, sort_keys=True)}", flush=True)

    def _set_error_payload(self, error_message: str, *, metadata: Optional[dict[str, Any]] = None) -> None:
        next_metadata = dict(metadata or {})
        next_metadata.setdefault("status", "error")
        next_metadata.setdefault("error", error_message)
        self._cached_payloads = {
            "pose_summary": [],
            "confidence_summary": {},
            "structure_artifacts": {},
            "run_metadata": next_metadata,
        }

    def _emit_outputs(self, t: float) -> None:
        self._outputs = {}
        for name in self.outputs():
            self._outputs[name] = _make_signal(source="diffdock", name=name, value=self._cached_payloads.get(name, {}), emitted_at=t, spec=self.outputs().get(name))

    def _confidence_band(self, value: Any) -> str:
        if not isinstance(value, (int, float)):
            return "unknown"
        if value > 0:
            return "high"
        if value > -1.5:
            return "moderate"
        return "low"

    def _structure_artifact_id(self, path: Path) -> str:
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        return f"structure-{digest}"
