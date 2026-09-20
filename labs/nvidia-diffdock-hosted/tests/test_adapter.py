"""Synthetic technical fixtures only; no mock response is inference evidence."""
import copy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
import yaml
from rdkit import Chem
from biosim import BioModule, BioWorld, ExecutionContext, ExecutionPolicy, SignalSpec, make_signal
from biosim.pack import validate_lab_source

LAB = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('nvidia_diffdock', LAB/'owned/models/main/model.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
CTX = ExecutionContext(policy=ExecutionPolicy.ONCE_BEFORE_RUN, run_start=0, run_end=1)
RID = '12345678-1234-5678-1234-567812345678'


def sdf(smiles='CCO'):
    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (i * 1.4, float(i % 2), 0.0))
    mol.AddConformer(conf)
    return Chem.MolToMolBlock(mol) + '\n$$$$\n'


@pytest.fixture
def scientific_inputs():
    return {'receptor': {'receptor_id': 'synthetic-receptor', 'pdb_text': 'ATOM      1  N   ALA A   1       0.000   1.000   2.000  1.00 20.00           N  \nATOM      2  CA  ALA A   1       1.000   1.000   2.000  1.00 20.00           C  \n', 'preparation': {'purpose': 'synthetic technical input only'}},
            'ligand': {'ligand_id': 'synthetic-ligand', 'sdf_text': sdf(), 'preparation': {'purpose': 'synthetic technical input only'}},
            'run_options': {'parameters': {'num_poses': 1}}}


def request(values):
    return m.prepare_request(values['receptor'], values['ligand'], values['run_options'])


def native(req):
    return {'status': 'success', 'details': 'synthetic transport test', 'protein': req['protein'], 'ligand': req['ligand'], 'ligand_positions': [req['ligand']] * req['num_poses'], 'position_confidence': [-0.98] * req['num_poses'], 'trajectory': [''] * req['num_poses']}


def signals(values):
    ports = m.NvidiaDiffDock().inputs()
    return {k: make_signal(spec=ports[k], value=v, source='synthetic-fixture', name=k, emitted_at=0.0) for k, v in values.items()}


def transport(monkeypatch, handler):
    real = httpx.Client
    monkeypatch.setattr(m.httpx, 'Client', lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setenv('NVIDIA_API_KEY', 'unit-key')
    monkeypatch.setattr(m.time, 'sleep', lambda _: None)


def test_manifest_runtime_parity():
    module = m.NvidiaDiffDock()
    manifest = yaml.safe_load((LAB/'owned/models/main/model.yaml').read_text())
    lab = yaml.safe_load((LAB/'lab.yaml').read_text())
    for direction, ports in [('inputs', module.inputs()), ('outputs', module.outputs())]:
        declared = {p['name']: p for p in manifest['io'][direction]}
        assert set(ports) == set(declared)
        assert {p['maps_to'] for p in lab['io'][direction]} == {'main.' + p for p in ports}
        for name, spec in ports.items():
            p = declared[name]
            for attr in ['signal_type', 'schema', 'dtype', 'emitted_unit', 'required', 'default']:
                assert p.get(attr) == getattr(spec, attr)
            assert tuple(p.get('accepted_units', ())) == (spec.accepted_units or ())
    result = validate_lab_source(LAB)
    assert result.valid, result.errors
    assert len(lab['models']) == 1
    assert module.execution_policy == ExecutionPolicy.ONCE_BEFORE_RUN


@pytest.mark.parametrize('option,value', [('num_poses', 0), ('num_poses', 101), ('num_poses', True), ('time_divisions', 2), ('time_divisions', 21), ('steps', 0), ('steps', 19), ('steps', 1.2), ('save_trajectory', True), ('skip_gen_conformer', True), ('is_staged', True), ('seed', 7), ('steps', None)])
def test_rejects_unsupported_options(scientific_inputs, option, value):
    scientific_inputs['run_options']['parameters'][option] = value
    with pytest.raises(ValueError):
        request(scientific_inputs)


@pytest.mark.parametrize('failure', ['two_ligands', 'invalid_ligand', 'empty_identity', 'no_preparation', 'duplicate_atom', 'hetero_atom', 'alternate_atom', 'nan_coordinate'])
def test_rejects_scientific_input_ambiguities(scientific_inputs, failure):
    receptor, ligand = scientific_inputs['receptor'], scientific_inputs['ligand']
    if failure == 'two_ligands': ligand['sdf_text'] += sdf('CC')
    elif failure == 'invalid_ligand': ligand['sdf_text'] = 'not SDF'
    elif failure == 'empty_identity': ligand['ligand_id'] = ''
    elif failure == 'no_preparation': receptor['preparation'] = {}
    elif failure == 'duplicate_atom': receptor['pdb_text'] += receptor['pdb_text'].splitlines()[0] + '\n'
    elif failure == 'hetero_atom': receptor['pdb_text'] += 'HETATM\n'
    elif failure == 'alternate_atom': receptor['pdb_text'] = receptor['pdb_text'][:16] + 'A' + receptor['pdb_text'][17:]
    elif failure == 'nan_coordinate': receptor['pdb_text'] = receptor['pdb_text'][:30] + '     nan' + receptor['pdb_text'][38:]
    with pytest.raises((ValueError, RuntimeError)):
        request(scientific_inputs)


@pytest.mark.parametrize('failure', ['status', 'echo', 'count', 'chemistry', 'confidence_nan', 'confidence_bool', 'unexpected_trajectory'])
def test_rejects_untrustworthy_outputs(scientific_inputs, failure):
    req = request(scientific_inputs)
    data = native(req)
    if failure == 'status': data['status'] = 'fail'
    elif failure == 'echo': data['protein'] += 'changed'
    elif failure == 'count': data['position_confidence'] = []
    elif failure == 'chemistry': data['ligand_positions'] = [sdf('CCN')]
    elif failure == 'confidence_nan': data['position_confidence'] = [float('nan')]
    elif failure == 'confidence_bool': data['position_confidence'] = [True]
    elif failure == 'unexpected_trajectory': data['trajectory'] = ['MODEL 1']
    with pytest.raises(ValueError):
        m.parse_response(json.dumps(data).encode(), req, scientific_inputs['receptor'], scientific_inputs['ligand'])


def test_all_poses_and_negative_scores_preserved(scientific_inputs):
    scientific_inputs['run_options']['parameters']['num_poses'] = 2
    req = request(scientific_inputs)
    data = native(req)
    raw = m.encode(data)
    output = m.parse_response(raw, req, scientific_inputs['receptor'], scientific_inputs['ligand'])
    assert len(output['docking_poses']['poses']) == 2
    assert [x['value'] for x in output['pose_confidence']['items']] == [-0.98, -0.98]
    assert output['native_response'].encode() == raw
    assert output['docking_poses']['poses'][0]['atom_index_base'] == 0
    assert output['docking_poses']['poses'][0]['coordinate_frame_receptor_sha256'] == m.sha(req['protein'].encode())


def test_missing_secret_fails_before_network(monkeypatch, scientific_inputs):
    monkeypatch.delenv('NVIDIA_API_KEY', raising=False)
    monkeypatch.setattr(m.httpx, 'Client', lambda **_: pytest.fail('network attempted'))
    with pytest.raises(ValueError, match='not provisioned'):
        m.NvidiaDiffDock().execute(signals(scientific_inputs), context=CTX)


def test_async_request_submits_once(monkeypatch, scientific_inputs):
    methods = []
    req = request(scientific_inputs)
    def handler(r):
        methods.append(r.method)
        if r.method == 'POST':
            return httpx.Response(202, headers={'nvcf-reqid': RID}, json={'status': 'accepted'})
        assert str(r.url) == m.POLL_ENDPOINT + RID
        return httpx.Response(200, headers={'nvcf-reqid': RID}, json=native(req))
    transport(monkeypatch, handler)
    module = m.NvidiaDiffDock()
    result = module.execute(signals(scientific_inputs), context=CTX)
    assert methods == ['POST', 'GET']
    assert result['run_provenance']['provider']['request_id'] == RID
    assert 'unit-key' not in json.dumps(result)


def test_resume_validates_digest_and_uses_get(monkeypatch, scientific_inputs):
    req = request(scientific_inputs)
    calls = []
    def handler(r):
        calls.append(r.method)
        return httpx.Response(200, headers={'nvcf-reqid': RID}, json=native(req))
    transport(monkeypatch, handler)
    module = m.NvidiaDiffDock(resume_request_id=RID, resume_request_sha256='wrong')
    with pytest.raises(ValueError, match='digest differs'):
        module.execute(signals(scientific_inputs), context=CTX)
    assert calls == []
    module = m.NvidiaDiffDock(resume_request_id=RID, resume_request_sha256=m.sha(m.encode(req)))
    assert module.execute(signals(scientific_inputs), context=CTX)['run_provenance']['provider']['resumed']
    assert calls == ['GET']


@pytest.mark.parametrize('kind', ['http_error', 'oversized', 'redirect'])
def test_failures_never_retry_or_leak(monkeypatch, scientific_inputs, kind):
    calls = []
    def handler(r):
        calls.append(r.method)
        if kind == 'http_error': return httpx.Response(500, text='unit-key private error')
        if kind == 'redirect': return httpx.Response(302, headers={'location': 'https://untrusted.invalid/'})
        return httpx.Response(200, text='x' * 2048)
    transport(monkeypatch, handler)
    module = m.NvidiaDiffDock(max_response_bytes=1024)
    with pytest.raises(RuntimeError) as error:
        module.execute(signals(scientific_inputs), context=CTX)
    assert 'unit-key' not in str(error.value)
    assert 'unit-key' not in json.dumps(module.snapshot())
    assert calls == ['POST']


def test_bioworld_captures_all_terminal_outputs(monkeypatch, scientific_inputs):
    req = request(scientific_inputs)
    calls = []
    def handler(r):
        calls.append(r.method)
        return httpx.Response(200, headers={'nvcf-reqid': RID}, json=native(req))
    transport(monkeypatch, handler)
    module = m.NvidiaDiffDock()
    class InputFixture(BioModule):
        execution_policy = ExecutionPolicy.ONCE_BEFORE_RUN
        def outputs(self):
            return {name: SignalSpec.record(schema=schema, emitted_unit='1') for name, schema in [('receptor', m.RECEPTOR), ('ligand', m.LIGAND), ('run_options', m.OPTIONS)]}
        def execute(self, inputs, *, context):
            return scientific_inputs
    world = BioWorld(communication_step=1.0)
    world.add_biomodule('fixture', InputFixture())
    world.add_biomodule('main', module)
    for name in scientific_inputs:
        world.connect('fixture.' + name, 'main.' + name)
    world.run(duration=3.0)
    assert calls == ['POST']
    assert set(module.get_outputs()) == set(module.outputs())
    assert all(s.emitted_at == 0 for s in module.get_outputs().values())
    assert module.get_outputs()['docking_poses'].spec.emitted_unit == 'angstrom'
