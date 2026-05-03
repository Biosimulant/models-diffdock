#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import yaml


def _load_biosim_repo_paths(root: Path) -> None:
    monorepo = root.parents[2]
    biosim_src = monorepo / "biosim" / "src"
    if str(biosim_src) not in sys.path:
        sys.path.insert(0, str(biosim_src))


def _load_config(config_path: Path) -> dict:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"expected mapping config in {config_path}")
    return loaded


def main() -> int:
    root = Path(__file__).resolve().parent
    _load_biosim_repo_paths(root)

    parser = argparse.ArgumentParser(description="Run a real models-diffdock example directly.")
    parser.add_argument(
        "example",
        nargs="?",
        default="diffdock-minimal",
        choices=["diffdock-minimal"],
        help="Example folder to run.",
    )
    parser.add_argument("--config", type=Path, help="Explicit config path. Overrides example selection.")
    parser.add_argument("--work-dir", type=Path, help="Override module work_dir.")
    parser.add_argument("--runtime-dir", type=Path, help="Override managed runtime directory.")
    parser.add_argument("--output-json", type=Path, help="Optional path to write the final BioSignal payloads as JSON.")
    args = parser.parse_args()

    config_path = args.config.resolve() if args.config else (root / args.example / "config.yaml")
    config = _load_config(config_path)
    model_cfg = config["model"]
    model_path = model_cfg.get("path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("model.path is required")
    model_root = (config_path.parent / Path(model_path)).resolve()
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))
    class_path = model_cfg["class"]
    module_name, class_name = class_path.split(":", 1)
    module_cls = getattr(importlib.import_module(module_name), class_name)

    parameters = dict(model_cfg.get("parameters") or {})
    if args.work_dir is not None:
        parameters["work_dir"] = str(args.work_dir.resolve())
    else:
        parameters.setdefault("work_dir", str((config_path.parent / "runs").resolve()))
    if args.runtime_dir is not None:
        parameters["runtime_dir"] = str(args.runtime_dir.resolve())

    module = module_cls(**parameters)

    from biosim.signals import BioSignal

    inputs = {
        name: _make_signal(source="example", name=name, value=value, emitted_at=0.0, spec=None)
        for name, value in (model_cfg.get("inputs") or {}).items()
    }
    if inputs:
        module.set_inputs(inputs)

    module.advance_window(0.0, 0.01)
    outputs = {name: signal.to_dict() for name, signal in module.get_outputs().items()}

    payload = {
        "example": config.get("example_name", args.example),
        "config_path": str(config_path),
        "outputs": outputs,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    run_metadata = outputs.get("run_metadata", {}).get("value", {})
    return 0 if run_metadata.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


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
