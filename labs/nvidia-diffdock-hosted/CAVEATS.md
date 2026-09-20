# DiffDock limitations

This is an individually authored NVIDIA-hosted adapter, not upstream DiffDock-L.
The documented API/NIM package is 2.3.0, but actual hosted weight/image hashes are
unknown. NVIDIA performs model compute; local/Hub workers perform HTTP/parsing.
The Hub worker's secure NVIDIA_API_KEY and dependency execution are unverified.

Initial interface covers one SDF ligand and one explicitly prepared protein-only
PDB. No implicit cofactor deletion, protonation adjustment, alternate-conformer
selection, staged assets, trajectories, SMILES batches or skipped conformer
generation. Those modes need separately reviewed interfaces and tests.

NVIDIA may remove explicit hydrogen atoms. Normalized molecular graph/charge/
stereo and heavy-atom mapping are checked; no all-atom preservation is claimed.
Coordinates and receptor echoes do not establish a correct binding pose.
Symmetric atom permutations require a proper docking metric beyond one graph
mapping. Native scores are neither affinity nor calibrated binding probability.

The NVIDIA 8G43/ZU6 example filters to ATOM records, deleting zinc and other
non-protein records. It is technical interface evidence only. It is not a
held-out benchmark or proof of accuracy in metal/cofactor-dependent systems.
Independent benchmark split, leakage review, tolerances and uncertainty remain
unresolved. Public qualification must not precede adequate claim-specific evidence.

Timeout only ends client observation. Use the safe request ID/digest to resume;
do not submit another POST automatically. A missing provider ID after a network
failure may prevent safe recovery. No run/release result is inferred from logs.
