# DiffDock-L Docking Predictor

`diffdock-diffdockl-docking-predictor` is a native `biosim.BioModule` wrapper
around upstream DiffDock-L inference for **single-complex** docking.

## Public Interface

Inputs:
- `protein_path`: path to a receptor `.pdb`
- `ligand_description`: either a ligand file path that RDKit can read or a SMILES string
- `run_options`: optional map with `complex_name`, `samples_per_complex`,
  `inference_steps`, `batch_size`, and `save_visualisation`

Outputs:
- `pose_summary`
- `confidence_summary`
- `structure_artifacts`
- `run_metadata`

## Runtime Model

Local default:
- `runtime_mode: managed`
- clones upstream DiffDock at tag `v1.1.3` into `.runtime/diffdock/repo`
- creates a repo-local venv under `.runtime/diffdock/venv`
- installs `requirements/runtime-gpu.txt`

Remote default:
- manifest-declared remote init overrides force `runtime_mode: managed`
- the wrapper creates its own venv under the mounted remote runtime/cache roots
  and installs `requirements/runtime-gpu.txt` itself
- the wrapper clones the pinned DiffDock repo and runs inference from that
  managed runtime
- `requirements/runtime-gpu.txt` intentionally uses a Python 3.11-compatible
  torch/PyG stack for the desktop remote sandbox instead of upstream's older
  Python 3.10-era pins

The first real run needs:
- internet access to clone the upstream repo and download DiffDock model assets
- a working `git` executable
- an NVIDIA GPU for release-grade validation

## Example Assets

The checked-in assets under `data/1a0q/` use the official upstream DiffDock
single-complex sample:

- `1a0q_protein_processed.pdb`
- `1a0q_ligand.sdf`

The example configs keep using the official sample CSV SMILES entry because it
exercises the most portable remote path, but the wrapper also accepts the
checked-in ligand file directly.
