# models-diffdock

> Storage-only repo: each former root model now lives in `labs/<slug>/model/` and is wrapped by
> `labs/<slug>/lab.yaml`. This repo has no repo-level import catalog and no composed labs at the root.

Curated collection of **DiffDock-family docking models** for the **biosim**
platform.

This repository currently ships one native Python `biosim.BioModule` wrapper:
`diffdock-diffdockl-docking-predictor`, a single-complex DiffDock-L docking
module for protein PDB plus ligand runs.

## What's Inside

### Wrapper Sublabs

| Sublab | Description |
|---|---|
| `diffdock-diffdockl-docking-predictor` | Native DiffDock-L wrapper for single receptor PDB and single ligand docking workflows. |

## Scope

This repository is for:
- native DiffDock-family wrappers that implement the `biosim.BioModule` contract
- single-complex docking runs that emit compact summaries plus file-backed structural artifacts
- portable examples that can be exported to `.bsilab` and validated against remote GPU execution

This repository is not for:
- sequence-first receptor folding workflows
- GNINA rescoring or post-processing
- unrelated docking engines

## Remote Execution

The DiffDock model uses the existing generic remote execution path:

- the wrapper bootstraps from the checked-in `requirements/runtime-gpu.txt`
- remote runs force `runtime_mode: managed` through manifest-declared remote init overrides
- the wrapper creates its managed runtime under the mounted remote cache/work roots, avoiding current Hub-side requirements parsing limits for pip option lines
- the wrapper clones the pinned upstream DiffDock repo at `v1.1.3`, runs inference, and emits a merged `top_rank_complex.pdb` artifact for the existing `structure3d` renderer

The release-grade validation target is Linux + NVIDIA GPU on Modal.

## Examples

See [examples/README.md](examples/README.md) for the example inventory, including
the remote `.bsilab` builder used for desktop end-to-end validation.
