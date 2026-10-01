# DiffDock-L single-complex pose exploration

This Lab generates candidate ligand poses for one prepared receptor PDB and one
ligand (SMILES or a supported molecular file). It reports a within-complex pose
ranking and inspectable SDF/PDB artifacts. It does not predict binding affinity,
provide calibrated binding probabilities, prepare raw receptors, or establish
experimental activity.

## Research status

The repaired 1.1.0 scientific revision passed an exact managed Modal T4 run on
1 October 2026 with twenty inference steps, two samples and batch size two.
Two SDF poses and a 6,294-atom complex were durably delivered. All seven result
files and their signed Passport bindings were independently verified. The first
pose passes the basic overlap screen; the second is rejected for six receptor–
ligand heavy-atom contacts below 1.5 angstrom. Its raw coordinates and rank remain
available for inspection, with the rejection visible in the table and report.

The executed Python/PyTorch/CUDA/package inventory and both observed ESM hashes
are recorded. This supports inspectable candidate-pose exploration, not
experimental binding or general docking accuracy. Passport remains REVIEW:
exact primary-source evidence is not pinned and transitive caches are not fully
locked. The bundled 1a0q receptor and example SMILES are a small execution
example, not an independent docking-accuracy benchmark. Historical screenshots
in `assets/` predate this repair and are not its validation evidence. This
documentation correction preserves the tested 1.1.0 science.

Verified run: `3187e7c7-a562-4cc2-a8a5-4887046fe2dd`, private revision
`e8cd84be-f0e3-4dd7-a276-f615f33fb759`, results SHA-256
`ab9585c86431bb78330b8fdbf4583191631361851149b9f890ee67383e0e1f43`.

## Inputs and controls

- `protein_path`: prepared receptor PDB; defaults to
  `data/1a0q/1a0q_protein_processed.pdb` relative to `models/core`.
- `ligand_description`: SMILES (including slash/backslash stereochemistry) or a
  supported ligand file. The example uses `COc(cc1)ccc1C#N`.
- `run_options`: a mapping merged onto the Lab preset. `complex_name` is a safe
  output name; `samples_per_complex`, `inference_steps`, and `batch_size` are
  positive integers; `save_visualisation` is boolean. Unknown keys are errors.

The Lab preset requests two samples, twenty inference steps, and batch size two.
It replaces the previous four-step smoke preset, whose managed diagnostic
produced no poses. One successful twenty-step diagnostic does not establish
that step count was the sole cause or validate docking accuracy. Upstream uses
ten samples and twenty steps; the two-sample Lab remains a bounded exploration,
not an accuracy benchmark. Sampling is stochastic; repeated runs are not
claimed to be byte-identical.

Upstream v1.1.3 loads YAML after command-line arguments. This wrapper writes one
`effective_inference_args.yaml` per run with the requested options and absolute
input/output/checkpoint paths, preventing YAML from silently restoring defaults.
`actual_steps` is the smaller of the upstream value (19) and requested inference
steps; all other upstream diffusion settings are preserved. The full effective
configuration and its checksum are included in run metadata.

## Runtime and reproducibility

Managed execution requires Linux and Python 3.11 for the pinned Torch 2.0.1/CUDA
11.7/PyG wheels. The manifest requests one NVIDIA GPU and Biosimulant 0.0.34.
Unsupported managed hosts fail before installing dependencies. External mode
requires a separately prepared compatible environment; an explicitly supplied
missing interpreter is an error.

Source is pinned to DiffDock v1.1.3 commit
`9a22cbcbc7612c7565c80e8399d9be298971f156`. Every new or reused checkout must match
and have no modified tracked files. The official v1.1 checkpoint archive is
pinned by SHA-256 (`5a95b6a1555be47ab1d6f0a8ffd25152f7fe32f5956005bb821e13e7a37d4a3d`),
as are its four checkpoint/parameter files in
`models/core/data/diffdock-checkpoints.json`. Modified caches fail closed.
Source/weights are MIT licensed by upstream. First execution needs network access
for source, dependencies, these checkpoints and ESM assets; offline readiness
requires all relevant caches and has not been established for this repair.

Input hashes, source revision, requirements hash, effective configuration,
checkpoint hashes, command, logs and explicit status accompany each result.
Transitive dependencies and ESM downloads are not fully locked by this wrapper.
The executed managed environment and observed ESM artifact hashes are recorded
in run metadata; hashes identify observed bytes rather than upstream authenticity
or deterministic reruns.

## Reading results

`pose_summary` lists contiguous ranked poses and raw confidence scores. Positive
scores use the upstream heuristic high band, scores above -1.5 through zero use
moderate, and scores at or below -1.5 use low. These are heuristics, not calibrated
probabilities. Comparisons across unrelated complexes or receptor conformations
are not established. See [upstream interpretation](https://github.com/gcorso/DiffDock/tree/9a22cbcbc7612c7565c80e8399d9be298971f156#faq).

Each pose includes a basic geometry screen: any receptor–ligand heavy-atom pair
closer than 1.5 angstrom rejects that pose for severe overlap. Passing this screen
does not establish binding or docking accuracy. Raw ranks and coordinates remain
available; rejection is visible in the pose table and complex annotation.

`confidence_summary` includes all scores, rejected ranks, the number passing the
basic geometry screen and the interpretation limit.
`structure_artifacts` contains the same ranked top pose used by the summary,
merged with the receptor for visualization. `run_metadata.status` is completed
only when the requested pose count, finite scores and expected artifacts are
present. Missing, malformed or incomplete outputs produce explicit error status
with empty scientific outputs. Inspect geometry and pursue independent physical
and experimental checks before relying on a proposed pose.

The core and presenter execute once before each BioWorld run, with dependencies
drained by Biosimulant's execution policy.

## Local checks

Install Biosimulant 0.0.34, PyYAML 6.0.2 and pytest in an isolated environment, then:

```sh
python -m pytest labs/diffdock-diffdockl-docking-predictor -q
biosimulant labs validate labs/diffdock-diffdockl-docking-predictor --json
biosimulant labs package labs/diffdock-diffdockl-docking-predictor --out dist/ --visibility private
```

The real smoke test is opt-in (`BIOSIM_DIFFDOCK_RUN_REAL_SMOKE=1`) on a supported
runner. Mock wrapper checks do not establish scientific docking performance.
