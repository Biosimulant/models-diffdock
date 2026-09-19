# BIO-002: NVIDIA DiffDock interface design

Draft, pending actual response inspection and requirements review. This document
is not executable implementation or approved scientific qualification.

## Invocation and exact-version boundary

Use a hand-authored CPU BioModule HTTP client, ExecutionPolicy.ONCE_BEFORE_RUN.
NVIDIA runs the GPU model. No upstream model substitution, auto-installing model
weights, replay-as-inference, or artificial biological time advancement.

The hosted page declares `/v1/molecular-docking/diffdock/generate`, while its
embedded 2.3.0 example uses `/v1/biology/mit/diffdock`. Resolve actual routing with
the native reference before selecting a fixed production endpoint. Do not infer
checkpoint identity from successful HTTP routing or documentation version alone.
Native reference 89803270-54d4-4710-8ba3-b3f166ce7698 established that the embedded
example route `/v1/biology/mit/diffdock` succeeds; use that fixed route.

## Proposed ports

All port definitions below remain proposals until the matching manifest,
SignalSpec methods, Lab mappings and tests exist. Top-level record schema fields
are explicit. Nested rows require model-specific checkers; no draft profile is
attached to a released runtime.

| Direction / port | Signal contract | Meaning and validation |
|---|---|---|
| Input receptor | record {receptor_id:str,pdb_text:str,preparation:json}, unit 1 | Mixed identity/artifact envelope. PDB coordinates angstrom. Parse unique chain/residue/insertion/atom/altloc identifiers; preserve original bytes and frame. Require stated preparation. |
| Input ligand | record {ligand_id:str,sdf_text:str,preparation:json}, unit 1 | Exactly one SDF molecule; explicit graph, charges, stereo and initial angstrom coordinates. Preserve input atom indices and identity. No automatic chemistry changes. |
| Input run_options | record {parameters:json}, unit 1 | Operational controls only; validated allowlist. No scientific compatibility profile. |
| Output docking_poses | record {receptor_id:str,ligand_id:str,poses:json,coordinate_unit:str,request_sha256:str}, unit angstrom | All pose SDFs, byte/hash/source-index and atom mapping. One receptor frame. Ranking and molecule identity explicit; geometry validation does not establish accuracy. |
| Output pose_confidence | record {ligand_id:str,items:json,score_semantics:str,request_sha256:str}, unit 1 | One finite native score per source pose, unitless model score. May be negative. No Boltz profile, affinity conversion, probability claim or invented confidence bands. |
| Output run_provenance | record {request:json,provider:json,inputs:json,environment:json,limitations:json}, unit 1 | Mixed operational report containing request/response hashes, safe headers, observed IDs/times and unknown backend identities as null. |
| Output native_response | scalar str, unit 1 | Exact UTF-8 JSON response text. Scientific consumers use extracted validated ports. |

`docking_poses` carries coordinates as its primary quantity; per-pose artifact
metadata is auxiliary. A format-only SDF profile cannot certify receptor frame,
chemical identity or scientifically meaningful ranking.

## Provider options from 2.3.0 OpenAPI

| Option | Documented type/bounds/default | Proposed Lab behavior |
|---|---|---|
| num_poses | integer 1..100, default 10 | Explicit integer; one-pose technical reference. |
| time_divisions | integer 3..20, default 20 | Algorithmic divisions, no biological units. |
| steps | integer 1..18, default 18 | Algorithmic denoising count, no biological units. |
| save_trajectory | boolean, default false | Initially require false until all trajectory outputs and algorithmic indices are declared/tested. |
| skip_gen_conformer | boolean, default false | Preserve supplied choice; true requires valid input conformer and separate coverage. |
| is_staged | boolean, default false | Require false; inline inputs only initially. |

The provider LigandFormat enum is mol2/sdf/txt, not a literal smiles enum. Its
schema permits nullable fields and has no required-field list; the Lab requires
actual receptor/ligand content and rejects nulls. Success response schema is
unspecified in OpenAPI. NVIDIA getting-started documents status/details/protein/
ligand/ligand_positions/position_confidence and optional trajectory. Actual hosted
capture must resolve any wrapper or field differences before implementation.

## Implementation and verification plan

Pin and test an available Biosimulant runtime independently of local source and
managed runtime versions. CPU parsing uses pinned httpx, Biopython and RDKit.
Set client observation below the managed worker cap, bounded response bytes and
one POST. Poll only a validated provider request UUID on the documented NVIDIA
status endpoint. Persist an ambiguous accepted request as recoverable, not failed
compute; do not submit a duplicate automatically.

Native reference uses the documented 8G43 ATOM-only PDB and ZU6 ideal SDF, one
pose, 20 divisions, 18 steps, no trajectory or staged data. Preserve source
files, exact filtering recipe, access time and hashes. No model weights needed.
Parse and compare actual returned chemistry with an independent RDKit check.
Establish transport/interface validity separately from docking performance.

The native response removes explicit hydrogens (36 input atoms, 21 output heavy
atoms in ZU6). Compare hydrogen-normalized molecular identity including charge
and stereochemistry; retain explicit source/output heavy-atom mappings and
hydrogen policy. Do not claim full input atom-index preservation. Disabled
trajectory returns a one-item list containing an empty string; preserve it in
the native response and do not create a fake trajectory artifact.

No managed credentials or bounds have been established for DiffDock. Exact
checkpoint/image versions, seed support, stochastic tolerances and scientific
benchmark acceptance remain open. Do not claim managed run or release success.

Sources: the NVIDIA API reference, getting-started and release notes linked in
MRS.md; local runtime interfaces and the existing compatibility catalog.
