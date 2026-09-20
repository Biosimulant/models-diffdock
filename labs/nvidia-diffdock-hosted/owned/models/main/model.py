"""BIO-002: individual NVIDIA DiffDock adapter; qualification remains separate."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone

import httpx
from Bio.PDB import PDBParser
from rdkit import Chem
from biosim import BioModule, ExecutionPolicy, SignalSpec

ENDPOINT = 'https://health.api.nvidia.com/v1/biology/mit/diffdock'
POLL_ENDPOINT = 'https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/'
RECEPTOR = {'receptor_id': 'str', 'pdb_text': 'str', 'preparation': 'json'}
LIGAND = {'ligand_id': 'str', 'sdf_text': 'str', 'preparation': 'json'}
OPTIONS = {'parameters': 'json'}
POSES = {'receptor_id': 'str', 'ligand_id': 'str', 'poses': 'json', 'coordinate_unit': 'str', 'request_sha256': 'str'}
CONFIDENCE = {'ligand_id': 'str', 'items': 'json', 'score_semantics': 'str', 'request_sha256': 'str'}
PROVENANCE = {'request': 'json', 'provider': 'json', 'inputs': 'json', 'environment': 'json', 'limitations': 'json'}
DEFAULTS = {'num_poses': 10, 'time_divisions': 20, 'steps': 18,
            'save_trajectory': False, 'skip_gen_conformer': False, 'is_staged': False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def record(value, schema, label):
    require(isinstance(value, dict) and set(value) == set(schema), label + ' fields must match declared schema')


def molecule(text):
    require(isinstance(text, str) and 0 < len(text.encode()) <= 1_000_000, 'SDF must be nonempty bounded text')
    molecules = list(Chem.ForwardSDMolSupplier(io.BytesIO(text.encode()), removeHs=False))
    require(len(molecules) == 1 and molecules[0] is not None, 'Exactly one valid SDF molecule required')
    mol = molecules[0]
    require(mol.GetNumHeavyAtoms() > 0 and mol.GetNumConformers() == 1, 'Ligand needs heavy atoms and one conformer')
    require(all(math.isfinite(float(c)) for xyz in mol.GetConformer().GetPositions() for c in xyz), 'Nonfinite ligand coordinates')
    return mol


def chemical_identity(mol):
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)


def prepare_request(receptor, ligand, options):
    record(receptor, RECEPTOR, 'receptor')
    record(ligand, LIGAND, 'ligand')
    record(options, OPTIONS, 'run_options')
    for obj, key in [(receptor, 'receptor_id'), (ligand, 'ligand_id')]:
        require(isinstance(obj[key], str) and 0 < len(obj[key]) <= 128, 'Explicit bounded identity required')
        require(isinstance(obj['preparation'], dict) and bool(obj['preparation']), 'Explicit preparation record required')
    pdb = receptor['pdb_text']
    require(isinstance(pdb, str) and 0 < len(pdb.encode()) <= 2_000_000, 'Receptor PDB must be nonempty bounded text')
    lines = pdb.splitlines()
    require(not any(line.startswith(('HETATM', 'MODEL ', 'ENDMDL')) for line in lines), 'Initial adapter requires one explicitly prepared protein-only PDB, no HETATM or multimodel records')
    atoms = [line for line in lines if line.startswith('ATOM  ')]
    require(bool(atoms), 'Receptor contains no ATOM records')
    require(all(len(line) >= 54 and line[16] == ' ' for line in atoms), 'Resolve alternate conformers explicitly before docking')
    identifiers = [(line[21:27], line[12:16]) for line in atoms]
    require(len(set(identifiers)) == len(identifiers), 'Duplicate receptor atom identity')
    require(len({line[6:11] for line in atoms}) == len(atoms), 'Duplicate receptor atom serial')
    require(all(math.isfinite(float(line[a:b])) for line in atoms for a, b in [(30, 38), (38, 46), (46, 54)]), 'Nonfinite receptor coordinates')
    structure = PDBParser(PERMISSIVE=False, QUIET=True).get_structure('receptor', io.StringIO(pdb))
    require(len(list(structure.get_atoms())) == len(atoms), 'Receptor parser dropped atoms')
    molecule(ligand['sdf_text'])
    parameters = options['parameters']
    require(isinstance(parameters, dict) and set(parameters) <= set(DEFAULTS), 'Unknown inference option')
    chosen = {**DEFAULTS, **parameters}
    for key, lower, upper in [('num_poses', 1, 100), ('time_divisions', 3, 20), ('steps', 1, 18)]:
        require(type(chosen[key]) is int and lower <= chosen[key] <= upper, 'Invalid integer option: ' + key)
    for key in ['save_trajectory', 'skip_gen_conformer', 'is_staged']:
        require(type(chosen[key]) is bool, 'Invalid boolean option: ' + key)
    require(not chosen['save_trajectory'] and not chosen['is_staged'], 'Trajectory and staged mode need separate reviewed contracts')
    require(not chosen['skip_gen_conformer'], 'Skipping conformer generation is not covered by the initial implementation')
    body = {'protein': pdb, 'ligand': ligand['sdf_text'], 'ligand_file_type': 'sdf', **chosen}
    encode(body)
    encode(receptor['preparation'])
    encode(ligand['preparation'])
    return body


def parse_response(raw, request, receptor, ligand):
    data = json.loads(raw)
    encode(data)  # Reject nonfinite values even in otherwise unused native fields.
    require(isinstance(data, dict) and data.get('status') == 'success', 'Provider response does not report success')
    require(data.get('protein') == request['protein'] and data.get('ligand') == request['ligand'], 'Provider input echo differs')
    sdf_values, confidence = data.get('ligand_positions'), data.get('position_confidence')
    require(isinstance(sdf_values, list) and isinstance(confidence, list), 'Missing native pose/confidence lists')
    require(len(sdf_values) == len(confidence) == request['num_poses'], 'Pose count/confidence mapping mismatch')
    require(data.get('trajectory') in (None, [], [''] * len(sdf_values)), 'Unexpected trajectory output when disabled')
    source = molecule(request['ligand'])
    source_heavy = Chem.RemoveHs(source)
    source_indices = [atom.GetIdx() for atom in source.GetAtoms() if atom.GetAtomicNum() != 1]
    poses, scores = [], []
    for index, (text, score) in enumerate(zip(sdf_values, confidence)):
        require(type(score) in (int, float) and math.isfinite(score), 'Confidence must be finite numeric score')
        mol = molecule(text)
        require(chemical_identity(mol) == chemical_identity(source), 'Output ligand chemistry, charge or stereochemistry differs')
        heavy = Chem.RemoveHs(mol)
        mapping = heavy.GetSubstructMatch(source_heavy, useChirality=True)
        require(len(mapping) == source_heavy.GetNumAtoms() == heavy.GetNumAtoms(), 'Heavy atom mapping incomplete')
        output_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        require(len(source_indices) == len(mapping) == len(output_indices), 'Unsupported isotope/hydrogen mapping')
        atom_map = [{'input_atom_index': source_indices[i], 'output_atom_index': output_indices[j]} for i, j in enumerate(mapping)]
        pose_id = ligand['ligand_id'] + ':pose:' + str(index)
        artifact = text.encode('utf-8')
        poses.append({'pose_id': pose_id, 'source_index': index, 'sdf_text': text, 'sha256': sha(artifact), 'bytes': len(artifact), 'coordinate_frame_receptor_sha256': sha(request['protein'].encode()), 'atom_index_base': 0, 'heavy_atom_mapping': atom_map, 'hydrogen_policy': 'Compare hydrogen-normalized identity; explicit H output may differ', 'input_explicit_atoms': source.GetNumAtoms(), 'output_explicit_atoms': mol.GetNumAtoms(), 'canonical_isomeric_smiles': chemical_identity(mol)})
        scores.append({'pose_id': pose_id, 'source_index': index, 'value': score})
    request_sha = sha(encode(request))
    return {
        'docking_poses': {'receptor_id': receptor['receptor_id'], 'ligand_id': ligand['ligand_id'], 'poses': poses, 'coordinate_unit': 'angstrom', 'request_sha256': request_sha},
        'pose_confidence': {'ligand_id': ligand['ligand_id'], 'items': scores, 'score_semantics': 'Native DiffDock unitless pose confidence; not binding probability or affinity', 'request_sha256': request_sha},
        'native_response': raw.decode('utf-8'),
    }


class NvidiaDiffDock(BioModule):
    execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN

    def __init__(self, observation_timeout_s=240.0, max_response_bytes=5_000_000,
                 resume_request_id=None, resume_request_sha256=None):
        require(isinstance(observation_timeout_s, (int, float)) and math.isfinite(observation_timeout_s) and 1 <= observation_timeout_s <= 1200, 'Invalid observation timeout')
        require(type(max_response_bytes) is int and 1024 <= max_response_bytes <= 20_000_000, 'Invalid response limit')
        require((resume_request_id is None) == (resume_request_sha256 is None), 'Resume requires provider ID and exact request digest')
        self.observation_timeout_s = float(observation_timeout_s)
        self.max_response_bytes = max_response_bytes
        self.resume_request_id = str(uuid.UUID(resume_request_id)) if resume_request_id else None
        self.resume_request_sha256 = resume_request_sha256
        self.state = {'status': 'not_started'}

    def inputs(self):
        return {'receptor': SignalSpec.record(schema=RECEPTOR, accepted_units=('1',), required=True),
                'ligand': SignalSpec.record(schema=LIGAND, accepted_units=('1',), required=True),
                'run_options': SignalSpec.record(schema=OPTIONS, accepted_units=('1',), required=False, default={'parameters': {}})}

    def outputs(self):
        return {'docking_poses': SignalSpec.record(schema=POSES, emitted_unit='angstrom'),
                'pose_confidence': SignalSpec.record(schema=CONFIDENCE, emitted_unit='1'),
                'run_provenance': SignalSpec.record(schema=PROVENANCE, emitted_unit='1'),
                'native_response': SignalSpec.scalar(dtype='str', emitted_unit='1')}

    def snapshot(self):
        return dict(self.state)

    def execute(self, inputs, *, context):
        require('receptor' in inputs and 'ligand' in inputs, 'Required scientific inputs missing')
        receptor, ligand = inputs['receptor'].value, inputs['ligand'].value
        options = inputs['run_options'].value if 'run_options' in inputs else {'parameters': {}}
        request = prepare_request(receptor, ligand, options)
        body = encode(request)
        request_sha = sha(body)
        require(self.resume_request_id is None or self.resume_request_sha256 == request_sha, 'Resume request digest differs')
        key = os.environ.get('NVIDIA_API_KEY')
        require(isinstance(key, str) and bool(key.strip()), 'Secure NVIDIA_API_KEY is not provisioned on this worker')
        start = time.monotonic()
        deadline = start + self.observation_timeout_s
        self.state = {'status': 'observing', 'request_sha256': request_sha, 'provider_request_id': self.resume_request_id}
        method = 'GET' if self.resume_request_id else 'POST'
        url = POLL_ENDPOINT + self.resume_request_id if self.resume_request_id else ENDPOINT
        raw, safe_headers = b'', {}
        try:
            with httpx.Client(follow_redirects=False, trust_env=False) as client:
                while True:
                    remaining = deadline - time.monotonic()
                    require(remaining > 0, 'Provider observation deadline reached; inspect request ID before retry')
                    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json', 'NVCF-POLL-SECONDS': str(min(30, max(1, int(remaining) - 1)))}
                    with client.stream(method, url, headers=headers, content=body if method == 'POST' else None, timeout=min(remaining, 60)) as response:
                        found_id = response.headers.get('nvcf-reqid') or response.headers.get('nvcf-request-id')
                        if found_id:
                            found_id = str(uuid.UUID(found_id))
                            require(self.state['provider_request_id'] in (None, found_id), 'Provider request identity changed')
                            self.state['provider_request_id'] = found_id
                        safe_headers = {k: v for k, v in response.headers.items() if k.lower() in {'date', 'content-type', 'nvcf-reqid', 'nvcf-request-id'}}
                        raw = b''
                        for chunk in response.iter_bytes():
                            require(len(raw) + len(chunk) <= self.max_response_bytes, 'Provider response exceeds size bound')
                            require(time.monotonic() < deadline, 'Provider observation deadline reached')
                            raw += chunk
                        status = response.status_code
                    if status == 202:
                        require(self.state['provider_request_id'] is not None, 'Accepted request lacks a recoverable ID')
                        method, url = 'GET', POLL_ENDPOINT + self.state['provider_request_id']
                        time.sleep(min(1, max(0, deadline - time.monotonic())))
                        continue
                    require(status == 200, 'NVIDIA response HTTP ' + str(status) + '; no automatic resubmission')
                    require(key.encode() not in raw, 'Unexpected credential in provider content')
                    break
            outputs = parse_response(raw, request, receptor, ligand)
        except Exception:
            self.state['status'] = 'observation_or_contract_failed'
            raise RuntimeError('DiffDock observation/contract failed; inspect safe snapshot request identity, never automatically resubmit') from None
        self.state['status'] = 'completed'
        outputs['run_provenance'] = {'request': {'body': request, 'sha256': request_sha},
            'provider': {'endpoint': ENDPOINT, 'request_id': self.state['provider_request_id'], 'response_sha256': sha(raw), 'response_bytes': len(raw), 'safe_headers': safe_headers, 'observed_at': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.monotonic() - start, 'resumed': self.resume_request_id is not None},
            'inputs': {'receptor': receptor, 'ligand': ligand},
            'environment': {'documented_api_version': '2.3.0', 'checkpoint_sha256': None, 'container_digest': None, 'provider_gpu': None, 'provider_seed': None},
            'limitations': ['Research docking; confidence is not affinity or probability.', 'Exact hosted checkpoint, container and GPU identities are unexposed.', 'Hydrogen-normalized graph identity does not establish pose accuracy.', 'Only single SDF ligand and prepared protein-only PDB are covered initially.', 'Scientific benchmark thresholds and qualification remain unresolved.']}
        return outputs
