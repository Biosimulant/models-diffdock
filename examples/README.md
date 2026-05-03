# DiffDock Examples

This repository currently includes:

- `diffdock-minimal`:
  Minimal single-complex example using the official `1a0q` processed receptor
  and the official example SMILES from DiffDock's sample CSV.
- `models/diffdock-diffdockl-docking-predictor/data/1a0q`:
  Checked-in official sample assets for both the processed receptor PDB and the
  ligand SDF.
- `diffdock-wiring`:
  Single-model `lab.yaml` example showing how to wire the model into a lab.
- `build_diffdock_bsispace.py`:
  Desktop automation script that exports a portable `.bsilab`, reimports it,
  stages it to Hub, runs it remotely on a GPU, and validates the synced results.
