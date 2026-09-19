# BIO-002: NVIDIA-hosted DiffDock requirements

Draft, 2026-09-20. No scientific acceptance approval or qualified release.

## Intended use

One Lab invokes NVIDIA's listed DiffDock service to generate candidate ligand
poses against one prepared receptor. It preserves all poses, native confidence
scores, receptor coordinate frame and request provenance. This is research
docking. Confidence is neither affinity nor calibrated binding probability.
Diffusion iteration is an algorithmic sampling axis, not biological time.

## Required behavior

- D01: Require receptor identity and explicit PDB text, ligand identity and
  explicit molecular representation. Record every preparation step. Never
  silently remove cofactors, change protonation or select an alternate atom.
- D02: Require one ligand per invocation initially. Reject unsupported batch
  input rather than lose ligand-to-output mapping. SDF is the initial format;
  MOL2/SMILES support needs its own contract coverage before being enabled.
- D03: Validate documented integer/bool options. Reject unknown keys and nulls
  in the Lab interface even where the provider schema is permissive. Defaults
  must be documented and copied into request provenance. Do not invent a seed.
- D04: Submit once to a documented NVIDIA HTTPS endpoint using the worker's
  secure NVIDIA_API_KEY. Retain asynchronous request identity. Never expose
  credentials, POST automatically after an ambiguous timeout, or follow an
  arbitrary credential-bearing redirect.
- D05: Preserve the native response and all returned ligand poses. Declare
  scientific quantities separately from file format. Use portable inline text
  plus SHA-256 and byte count, never a worker-local path as the deliverable.
- D06: Require a bijection between pose entries and confidence entries. Preserve
  native order and explicit source index; derive ranking only with a documented
  rule and no confidence bands borrowed from another release.
- D07: Parse every returned SDF and check finite coordinates, molecular graph,
  charge and stereochemistry against the submitted ligand with explicit atom
  mapping. Record any unestablished invariant as a failure or qualification gap.
- D08: Keep original receptor coordinates available so a consumer can interpret
  ligand coordinates in the same frame. Do not align ligand poses independently
  when measuring docking RMSD against a crystallographic reference.
- D09: Canonical io, Python SignalSpec and Lab output mappings must agree.
  Execute once before the run. Pin dependencies and bound response size/time.
- D10: Private managed validation must yield independently checked nonempty
  terminal artifacts and a reviewed Passport. Public publication needs approval
  of the exact release and adequate evidence for each stated claim.

## Acceptance and evidence gaps

Technical checks: malformed input, single-ligand enforcement, option bounds,
safe error/polling semantics, runtime/manifest parity, full pose capture,
confidence length/finite values, SDF graph/charge/stereo and frame preservation.
Synthetic responses can test parsing only. Real provider and managed executions
are separate evidence. No numerical accuracy threshold has been approved.

Scientific qualification remains blocked by an independently justified benchmark
split, PLINDER/SAIR training-overlap review, symmetry-corrected heavy-atom RMSD,
top-k success and physical-validity checks, uncertainty and repeatability limits.
NVIDIA's documentation example 8G43/ZU6 is a technical fixture, not held-out proof.
Its ATOM-only preparation drops zinc/cofactors/waters and lacks an independently
reviewed protonation/alternate-location protocol. Preserve that limitation.

## Identity and rights

NVIDIA NIM package 2.3.0 is documented. Its release notes say improved checkpoints
were integrated in 2.2.0; they do not provide weight identity equivalence. The
model-card description says v2.2.0 while its version section says v2.3. Hosted
OpenAPI info/version and x-nvai-meta/version say 2.3.0. These are documentation
versions, not observed checkpoint/container digests. The existing upstream
DiffDock-L v1.1.3 Lab is not an established equivalent. No weights redistributed.

Model card cites NVIDIA Open Model License and MIT. Hosted terms and container
terms require separate treatment. The API schema's MIT label alone does not
establish rights to all model artifacts or service deployment.

Sources:
- https://build.nvidia.com/mit/diffdock/modelcard
- https://docs.nvidia.com/nim/bionemo/diffdock/latest/release-notes.html
- https://docs.nvidia.com/nim/bionemo/diffdock/latest/getting-started.html
- https://docs.api.nvidia.com/nim/reference/mit-diffdock-infer
- https://www.rcsb.org/structure/8G43
