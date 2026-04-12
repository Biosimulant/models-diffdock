#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path("/Volumes/dem-ssd/imp/projects/Nitoons/Biosimulant")
DESKTOP_SRC_TAURI = REPO_ROOT / "bsim-platform" / "biosimulant-desktop" / "src-tauri"
DEFAULT_DESKTOP_BIN = DESKTOP_SRC_TAURI / "target" / "debug" / "biosimulant-desktop"
MODEL_DIR = REPO_ROOT / "models" / "models-diffdock" / "models" / "diffdock-diffdockl-docking-predictor"
EXAMPLE_CONFIG = REPO_ROOT / "models" / "models-diffdock" / "examples" / "diffdock-minimal" / "config.yaml"
DEFAULT_OUTPUT = Path("/Users/demi/Desktop/DiffDock_Remote_GPU_Example.bsispace")
_TRANSIENT_HUB_POLL_MARKERS = (
    "API error (503)",
    "no available server",
    "Network error:",
)
_REMOTE_SIZE_CAPACITY_MARKERS = (
    "API error (429)",
    "Concurrent run limit reached",
)


def parse_example_inputs(config_path: Path) -> tuple[str, str, dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model = config.get("model") if isinstance(config, dict) else {}
    inputs = model.get("inputs") if isinstance(model, dict) else {}
    protein_path = str(inputs.get("protein_path") or "").strip()
    ligand_description = str(inputs.get("ligand_description") or "").strip()
    run_options = inputs.get("run_options") if isinstance(inputs.get("run_options"), dict) else {}
    if not protein_path or not ligand_description:
        raise RuntimeError(f"Could not parse protein_path and ligand_description from {config_path}")
    return protein_path, ligand_description, dict(run_options)


def run_cli(
    desktop_bin: Path,
    data_dir: Path | None,
    command: str,
    payload: dict[str, Any],
) -> Any:
    cmd = [str(desktop_bin), "__biosimulant_cli__", "raw", command, "--json", "--input", json.dumps(payload)]
    if data_dir is not None:
        cmd.extend(["--data-dir", str(data_dir)])
    try:
        completed = subprocess.run(
            cmd,
            cwd=DESKTOP_SRC_TAURI,
            check=True,
            text=True,
            capture_output=True,
            env={
                **os.environ,
                **(
                    {"BIOSIMULANT_HUB_ACCESS_TOKEN": os.environ["BIOSIMULANT_HUB_ACCESS_TOKEN"]}
                    if os.environ.get("BIOSIMULANT_HUB_ACCESS_TOKEN")
                    else {}
                ),
            },
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"{command} failed: {detail or exc}") from exc
    envelope = json.loads(completed.stdout)
    if not envelope.get("ok", False):
        raise RuntimeError(f"{command} failed: {envelope}")
    return envelope["data"]


def _is_transient_hub_poll_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in _TRANSIENT_HUB_POLL_MARKERS)


def choose_remote_sizes(catalog: dict[str, Any], *, require_gpu: bool) -> list[dict[str, Any]]:
    sizes = catalog.get("sizes") or []
    credit_balance = float(catalog.get("credit_balance") or 0)
    affordable_sizes = [
        size for size in sizes if size.get("is_active") and float(size.get("credit_cost") or 0) <= credit_balance
    ]
    active_gpu_sizes = [
        size
        for size in affordable_sizes
        if size.get("is_active") and size.get("gpu_type") and (size.get("gpu_count") or 1) > 0
    ]
    if active_gpu_sizes:
        active_gpu_sizes.sort(
            key=lambda size: (
                0 if str(size.get("gpu_type", "")).upper() == "A10" else 1,
                size.get("credit_cost_per_minute") or 0,
                size.get("credit_cost") or 0,
            )
        )
        return active_gpu_sizes

    if require_gpu:
        raise RuntimeError("No affordable GPU remote sizes are available for this account.")
    raise RuntimeError("No affordable remote sizes are available for this account.")


def _is_remote_size_capacity_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in _REMOTE_SIZE_CAPACITY_MARKERS)


def create_remote_run_with_size_fallback(
    desktop_bin: Path,
    data_dir: Path,
    *,
    hub_space_id: str,
    space_commit: str,
    remote_sizes: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_error: Exception | None = None
    for remote_size in remote_sizes:
        try:
            remote_run = run_cli(
                desktop_bin,
                data_dir,
                "hub_create_remote_run",
                {
                    "payload": {
                        "space_id": hub_space_id,
                        "space_commit": space_commit,
                        "simulation_config": {"duration": 0.01, "tick_dt": 0.01, "initial_inputs": {}},
                        "remote_size_id": remote_size["id"],
                    }
                },
            )
            return remote_run, remote_size
        except RuntimeError as exc:
            last_error = exc
            if _is_remote_size_capacity_error(exc):
                continue
            raise
    raise RuntimeError(f"Could not create a remote run on any affordable GPU size: {last_error}")


def extract_structure_artifact(results_payload: dict[str, Any]) -> tuple[str, str | None]:
    visuals = results_payload.get("visuals") or []
    for module in visuals:
        if not isinstance(module, dict):
            continue
        for visual in module.get("visuals") or []:
            if not isinstance(visual, dict) or visual.get("render") != "structure3d":
                continue
            data = visual.get("data") or {}
            source = data.get("source") or {}
            artifact_id = source.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id.strip():
                file_name_hint = None
                source_path = source.get("path")
                if isinstance(source_path, str) and source_path.strip():
                    file_name_hint = Path(source_path).name
                return artifact_id, file_name_hint
    raise RuntimeError("Remote run results did not include a structure3d artifact.")


def assert_results_shape(results_payload: dict[str, Any]) -> None:
    outputs = results_payload.get("outputs")
    visuals = results_payload.get("visuals")
    if not isinstance(outputs, dict):
        raise RuntimeError("Run results are missing outputs.")
    for key in ("pose_summary", "confidence_summary", "structure_artifacts"):
        if key not in outputs:
            raise RuntimeError(f"Run outputs are missing `{key}`.")
    if not isinstance(visuals, list) or not visuals:
        raise RuntimeError("Run results are missing visuals.")
    extract_structure_artifact(results_payload)


def build_manifest(
    space_id: str,
    imported_model: dict[str, Any],
    protein_path: str,
    ligand_description: str,
    run_options: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "title": "DiffDock Remote GPU Example",
        "description": "Portable single-model DiffDock-L lab intended for remote GPU execution and inline 3D docked-complex visualisation.",
        "models": [
            {
                "alias": "diffdock",
                "path": imported_model["owned_path"],
                "provenance": {
                    "owner_space_id": space_id,
                    "owned_path": imported_model["owned_path"],
                    "imported_at": imported_model.get("imported_at"),
                    "local_revision": imported_model.get("local_revision"),
                    "dirty": False,
                },
                "parameters": {
                    "runtime_mode": "managed",
                    "default_protein_path": protein_path,
                    "default_ligand_description": ligand_description,
                    "default_run_options": run_options,
                },
            }
        ],
        "wiring": [],
        "runtime": {
            "duration": 0.01,
            "tick_dt": 0.01,
            "initial_inputs": {},
        },
        "scientific_context": {
            "question": "Can DiffDock-L predict a docked protein-ligand complex remotely and return a renderable 3D structure inside the desktop app?",
            "mode": "native-diffdock",
            "assumptions": [
                "Remote runs should use GPU-backed execution.",
                "The receptor is supplied as a prepared PDB file rather than a sequence-only fold request.",
            ],
            "expected_observables": [
                "Ranked pose summary values from DiffDock-L.",
                "A top-pose confidence score plus confidence band.",
                "A renderable structure3d visual backed by a persisted merged-complex artifact.",
            ],
        },
    }


def build_layout() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "diffdock", "type": "model", "x": 180, "y": 120},
        ],
        "viewport": {"x": 0, "y": 0, "zoom": 1},
    }


def mirror_and_wait_for_remote_results(
    desktop_bin: Path,
    data_dir: Path,
    imported_space_id: str,
    stage_result: dict[str, Any],
    remote_sizes: list[dict[str, Any]],
    timeout_seconds: int,
    poll_seconds: int,
) -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    remote_run, selected_remote_size = create_remote_run_with_size_fallback(
        desktop_bin,
        data_dir,
        hub_space_id=stage_result["hub_space_id"],
        space_commit=stage_result["space_commit"],
        remote_sizes=remote_sizes,
    )
    remote_run_id = str(remote_run.get("id") or "").strip()
    if not remote_run_id:
        raise RuntimeError(f"Remote run creation did not return an id: {remote_run}")
    remote_status = str(remote_run.get("status") or "queued")

    local_run = run_cli(
        desktop_bin,
        data_dir,
        "create_run",
        {
            "space_id": imported_space_id,
            "status": remote_status,
            "execution_target": "remote",
            "hub_run_id": remote_run_id,
            "simulation_config": {"duration": 0.01, "tick_dt": 0.01, "initial_inputs": {}},
        },
    )
    local_run_id = local_run["id"]

    deadline = time.time() + timeout_seconds
    last_remote_state: dict[str, Any] | None = None
    while time.time() < deadline:
        try:
            current = run_cli(
                desktop_bin,
                data_dir,
                "hub_get_remote_run",
                {"run_id": remote_run_id},
            )
        except RuntimeError as exc:
            if not _is_transient_hub_poll_error(exc):
                raise
            time.sleep(poll_seconds)
            continue
        last_remote_state = current
        status = str(current.get("status") or "")
        if status in {"completed", "failed", "cancelled"}:
            results_payload = run_cli(
                desktop_bin,
                data_dir,
                "hub_get_remote_run_results",
                {"run_id": remote_run_id},
            )
            run_cli(
                desktop_bin,
                data_dir,
                "sync_remote_run",
                {
                    "id": local_run_id,
                    "hub_run_id": remote_run_id,
                    "status": status,
                    "error_message": current.get("error_message"),
                    "duration_seconds": current.get("duration_seconds"),
                    "results_payload": results_payload,
                },
            )
            return local_run_id, remote_run_id, current, results_payload, selected_remote_size
        time.sleep(poll_seconds)
    raise TimeoutError(f"Remote run {remote_run_id} did not finish within {timeout_seconds} seconds. Last state: {last_remote_state}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate a portable DiffDock-L .bsispace package.")
    parser.add_argument("--desktop-bin", type=Path, default=DEFAULT_DESKTOP_BIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-remote-validation", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()

    protein_path, ligand_description, run_options = parse_example_inputs(EXAMPLE_CONFIG)
    build_root = Path(tempfile.mkdtemp(prefix="diffdock-bsispace-build-"))
    export_dir = build_root / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    package_tmp_path = export_dir / "DiffDock_Remote_GPU_Example.bsispace"

    data_dir = build_root / "data"
    import_data_dir = build_root / "imported-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    import_data_dir.mkdir(parents=True, exist_ok=True)

    try:
        created_space = run_cli(
            args.desktop_bin,
            data_dir,
            "create_space",
            {
                "title": "DiffDock Remote Example",
                "description": "Portable remote GPU DiffDock-L example lab",
            },
        )
        space_id = created_space["id"]
        imported_model = run_cli(
            args.desktop_bin,
            data_dir,
            "import_model_into_space_from_path",
            {
                "spaceId": space_id,
                "path": str(MODEL_DIR),
                "alias": "diffdock",
            },
        )
        run_cli(
            args.desktop_bin,
            data_dir,
            "save_space",
            {
                "id": space_id,
                "manifest": build_manifest(space_id, imported_model, protein_path, ligand_description, run_options),
                "wiringLayout": build_layout(),
            },
        )
        export_result = run_cli(
            args.desktop_bin,
            data_dir,
            "export_space_package",
            {"space_id": space_id, "output_path": str(package_tmp_path)},
        )
        if Path(export_result["path"]) != package_tmp_path:
            package_tmp_path = Path(export_result["path"])
        run_cli(
            args.desktop_bin,
            data_dir,
            "preview_package",
            {"package_path": str(package_tmp_path)},
        )
        imported_package = run_cli(
            args.desktop_bin,
            import_data_dir,
            "import_package",
            {"package_path": str(package_tmp_path)},
        )
        imported_space_id = (
            imported_package.get("local_space_id")
            or imported_package.get("space_id")
            or imported_package.get("local_id")
        )
        if not imported_space_id:
            raise RuntimeError(f"Could not determine imported space id from {imported_package}")

        validation_summary: dict[str, Any] | None = None
        if not args.skip_remote_validation:
            catalog = run_cli(
                args.desktop_bin,
                import_data_dir,
                "hub_get_remote_catalog",
                {},
            )
            gpu_sizes = choose_remote_sizes(catalog, require_gpu=True)
            stage_result = run_cli(
                args.desktop_bin,
                import_data_dir,
                "hub_stage_remote_space",
                {"space_id": imported_space_id},
            )
            local_run_id, remote_run_id, remote_state, results_payload, selected_remote_size = mirror_and_wait_for_remote_results(
                args.desktop_bin,
                import_data_dir,
                imported_space_id,
                stage_result,
                gpu_sizes,
                args.timeout_seconds,
                args.poll_seconds,
            )
            if remote_state.get("status") != "completed":
                raise RuntimeError(f"Remote run did not complete successfully: {remote_state}")
            assert_results_shape(results_payload)
            artifact_id, file_name_hint = extract_structure_artifact(results_payload)
            cached_artifact = run_cli(
                args.desktop_bin,
                import_data_dir,
                "hub_cache_remote_run_artifact",
                {
                    "run_id": remote_run_id,
                    "artifact_id": artifact_id,
                    "file_name_hint": file_name_hint,
                },
            )
            cached_path = Path(cached_artifact["local_path"])
            if not cached_path.is_file():
                raise RuntimeError(f"Desktop artifact cache did not produce a file at {cached_path}")
            local_results = run_cli(
                args.desktop_bin,
                import_data_dir,
                "get_run_results",
                {"run_id": local_run_id},
            )
            assert_results_shape(local_results)
            validation_summary = {
                "remote_size_id": selected_remote_size["id"],
                "remote_size_name": selected_remote_size["display_name"],
                "remote_run_id": remote_run_id,
                "local_run_id": local_run_id,
                "cached_structure_artifact": str(cached_path),
            }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(package_tmp_path.read_bytes())
        summary = {
            "output_package": str(args.output),
            "portable_space_package": str(package_tmp_path),
            "space_title": "DiffDock Remote Example",
            "model_source": str(MODEL_DIR),
            "remote_validation": validation_summary,
        }
        print(json.dumps(summary, indent=2))
    finally:
        pass


if __name__ == "__main__":
    main()
