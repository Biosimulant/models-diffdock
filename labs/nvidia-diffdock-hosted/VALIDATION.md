# BIO-002 validation evidence

37 local technical tests passed with Python 3.10, biosimulant 0.0.34, httpx
0.28.1, biopython 1.84, RDKit 2025.3.3, pyyaml 6.0.2 and pytest 8.3.5.
Manifest/runtime/Lab mappings and once-before-run BioWorld execution passed.
Tests cover input ambiguity, option bounds, all-pose capture, negative scores,
graph mismatch, input echo, polling, resume digest, missing secret, response
limits, errors and no automatic POST retries. Synthetic transport tests are
technical checks, not model inference or biological evidence.

## Native NVIDIA reference

2026-09-19 23:43:59–23:44:00 UTC, provider request
89803270-54d4-4710-8ba3-b3f166ce7698. Documented route
https://health.api.nvidia.com/v1/biology/mit/diffdock returned HTTP 200 and native
status success. One pose, 20 divisions, 18 steps, all mode flags false.

Request SHA-256: 371f6e974bcb10ca190311c5c939c3c55621f095de8cb9c847ee903d41d4b8ce.
Response: 69,212 bytes; SHA-256
996a9789aa646859f795d3696482d74378e26a598154d7b9b6af0139084bbd36.
Pose: 1,842 bytes; SHA-256
f39f3fff35636aa9dc95e85874e6ad6a5b54b669fe0906f4813b619a2550bfea.

Independent check: one finite score, matching normalized ligand graph/charge/
isomeric SMILES, 21 heavy atoms, finite coordinates and exact receptor/ligand
echoes. Disabled trajectory represented as [""]. Input SDF has 36 explicit atoms;
output removes hydrogen atoms. Native confidence 0.42948415875434875 is retained
without a probability or affinity interpretation.

## Fresh local adapter through BioWorld

The adapter first rejected unresolved alternate conformers before any network
request. For a separately prepared technical fixture, retained conformer A at
ASP A1135 (occupancy 0.70 vs B 0.30) and LEU A1162 (0.60 vs B 0.40), discarded
16 B records and blanked selected A markers. Original coordinate columns and
serial numbers are retained. Preparation, all selected serials and hashes are
in captured run provenance. This follows an explicitly documented highest-
occupancy choice; the model itself does not silently prepare a receptor.

2026-09-19 23:53:55–23:53:56 UTC, provider request
d4b01d7c-bf05-41a4-ad93-5f12aa6240a7. One fresh POST through the hand-authored
adapter and local BioWorld succeeded, with four terminal outputs at timestamp
0.0. Boundary time is computational scheduling, not biological time evolution.

Request SHA-256: 4bbff565e90276ef6aac92ce07446918f2dd02aaf463ab632d745bf928180693.
Source model.py SHA-256:
f2931a96dff5d515f1e6024b28f51de86fd039f39e431a3c27afcb02a17774b1.
Captured results: 212,308 bytes; SHA-256
760f5e24c44e6e87552067e403bd2d31eb3c3d96a7e69e5ffb8e7beb37e75a7f.
Native response: 67,900 bytes; SHA-256
d4d8124452bd4453a05616e39a7f5a2ae177acd22e5598bff1535d8956271294.
Pose SHA-256: c16e11b9905c7b347daac6fdac03a0e6bedb4ce350a052952aa53176feef174c.

An independent script, without importing the adapter, verified result/response/
pose bytes and hashes, all four output ports, native input echoes, native score
mapping, finite coordinates, heavy-atom mapping and graph/bond/charge identity.
Native score -2.2313895225524902 remains a model score. The prepared receptor
differs from the native example; these requests are NOT a same-input repeat.

## Outstanding gates

No managed Hub inference or new release occurred. Hub secure credential access,
installed runtime and actual managed artifacts remain unverified. Exact hosted
checkpoint/container/GPU/seed identity is undisclosed. An independent benchmark,
training-overlap audit, symmetry-corrected RMSD, suitable uncertainty and justified
acceptance thresholds remain outstanding. No broad accuracy claim follows.

Source for explicit conformer-selection convention:
https://biopython.org/docs/1.84/api/Bio.PDB.Atom.html
Scientific fixture: https://www.rcsb.org/structure/8G43
NVIDIA API: https://docs.api.nvidia.com/nim/reference/mit-diffdock-infer
