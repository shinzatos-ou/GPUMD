"""Grand canonical MC (`mc gcmc`): input checks, the sub-run driver, and the chemical-potential
limits of the insertion and deletion acceptance.

The 1024-atom diamond cell (the 16-atom fixture repeated 4 x 4 x 4) is about 16.5 A thick, so it is
not a small box for the 7 A radial cutoff of nep_C.txt. The run of 30 steps is split into three
sub-runs of 10 steps, each followed by one call of the trials.
"""
import re
import subprocess

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read

from conftest import MODELS_DIR, make_bulk_C

N_REAL = 1024
NUM_SWEEPS = 20
NUM_ACTIVE_CELLS = 1


def _write_atoms(path, atoms):
    cell = atoms.cell.array
    lattice = ' '.join(f'{x:.10f}' for x in cell.flatten())
    lines = [
        f'{len(atoms)}',
        f'Lattice="{lattice}" pbc="T T T" Properties=species:S:1:pos:R:3',
    ]
    for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions):
        lines.append(f'{symbol} {position[0]:.10f} {position[1]:.10f} {position[2]:.10f}')
    path.write_text('\n'.join(lines) + '\n')


def _write_model(path):
    atoms = make_bulk_C().repeat(4)
    assert len(atoms) == N_REAL
    _write_atoms(path, atoms)


def _make_cubic_model():
    return bulk('C', 'diamond', a=3.567, cubic=True).repeat(5)


def _write_cubic_model(path):
    _write_atoms(path, _make_cubic_model())


def _write_isolated_pair(path):
    lines = [
        '2',
        'Lattice="30 0 0 0 30 0 0 0 30" pbc="T T T" Properties=species:S:1:pos:R:3',
        'C 5 5 5',
        'C 20 20 20',
    ]
    path.write_text('\n'.join(lines) + '\n')


def _write_potential(path, energy_reference):
    potential = (MODELS_DIR / 'nep_C.txt').read_text()
    if energy_reference is not None:
        assert 'energy_reference 0\n' in potential
        potential = potential.replace(
            'energy_reference 0\n', f'energy_reference {energy_reference}\n'
        )
    path.write_text(potential)


def _run(tmp_path, gpumd_command, run_in, energy_reference=None, write_model=_write_model):
    _write_potential(tmp_path / 'nep_C.txt', energy_reference)
    write_model(tmp_path / 'model.xyz')
    (tmp_path / 'run.in').write_text(run_in)
    return subprocess.run([gpumd_command], cwd=tmp_path, capture_output=True, text=True)


def _read_mcmd(path):
    return np.loadtxt(path, ndmin=2)


def _read_auto_e0(stdout):
    match = re.search(r'auto isolated-atom energy e0 = (\S+) eV', stdout)
    assert match is not None, stdout
    return float(match.group(1))


def _gcmc_run_in(mc_line, ensemble='nvt_lan 300 300 100'):
    return (
        'potential nep_C.txt\n'
        'velocity 300 seed 11\n'
        'time_step 1\n'
        f'ensemble {ensemble}\n'
        f'{mc_line}\n'
        'dump_thermo 1\n'
        'dump_xyz 10 dump.xyz\n'
        'run 30\n'
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    'run_in, message',
    [
        (
            'potential nep_C.txt\n'
            'mc gcmc 10 300 300 1 1.0 1 1 C -7 12.011 auto\n'
            'run 10\n',
            'a_plus should be in (0, 1)',
        ),
        (
            'potential nep_C.txt\n'
            'mc gcmc 10 300 300 1 0.5 1 1 H -7 1.008 auto\n'
            'run 10\n',
            'a gcmc species is not allowed in the NEP model',
        ),
        (
            'potential nep_C.txt\n'
            'mc gcmc 10 300 300 1 0.5 1 1 C -7 12.011 auto unknown\n'
            'run 10\n',
            'unknown option for mc gcmc',
        ),
        (
            'potential nep_C.txt\n'
            'mc gcmc 10 300 300 1 0.5 1 1 C -7 12.011 auto overlap_cutoff 1 C 8.0\n'
            'run 10\n',
            'overlap_cutoff for mc gcmc should not exceed the NEP radial cutoff',
        ),
        (
            'potential nep_C.txt\n'
            'mc gcmc 10 300 300 1 0.5 1 1 C -7 12.011 auto parallel\n'
            'run 10\n',
            'n_sep of mc gcmc should be at least 2 with parallel',
        ),
        (
            'potential nep_C.txt\n'
            'mc gcmc 10 300 300 1 0.5 1 1 C -7 12.011 auto\n'
            'run 15\n',
            'should be a multiple of N_md',
        ),
        (
            'potential nep_C.txt\n'
            'mc gcmc 10 300 300 1 0.5 1 1 C -7 12.011 auto\n'
            'run 10\n'
            'run 10\n',
            'run should appear only once',
        ),
        (
            'potential nep_C.txt\n'
            'mc gcmc 10 300 300 1 0.5 1 1 C -7 12.011 auto\n'
            'run 10\n'
            'minimize sd 1.0e-6 10\n',
            'minimize should appear before run when mc gcmc is used',
        ),
        (
            'potential nep_C.txt\n'
            'mc gcmc 10 300 300 1 0.5 1 1 C -7 12.011 auto\n'
            'deposit 10 2 5.0 atom C 1 0\n'
            'run 10\n',
            'cannot be used together with deposit',
        ),
        (
            'potential nep_C.txt\n'
            'mc gcmc 10 300 300 1 0.5 1 1 C -7 12.011 auto\n'
            'dump_observer average 10 10 0 0\n'
            'run 10\n',
            'mc gcmc cannot be used together with dump_observer average',
        ),
        (
            'potential nep_C.txt\n'
            'mc gcmc 10 300 300 1 0.5 1 1 C -7 12.011 auto\n'
            'mc canonical 10 10 300 300\n'
            'run 10\n',
            'mc gcmc should be the only mc keyword',
        ),
    ],
)
def test_gcmc_rejects_invalid_input(tmp_path, gpumd_command, run_in, message):
    result = _run(tmp_path, gpumd_command, run_in)
    assert result.returncode != 0, result.stdout + result.stderr
    assert message in result.stdout + result.stderr


def test_gcmc_splits_the_run_into_sub_runs(tmp_path, gpumd_command):
    run_in = _gcmc_run_in('mc gcmc 10 300 300 1 0.5 1 1 C -100 12.011 auto seed 3')
    result = _run(tmp_path, gpumd_command, run_in)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('Running sub-run') == 3

    thermo = np.loadtxt(tmp_path / 'thermo.out', ndmin=2)
    assert thermo.shape[0] == 30
    assert np.all(np.isfinite(thermo))
    assert 270.0 < thermo[0, 0] < 330.0

    assert (tmp_path / 'run.in.original').read_text() == run_in
    assert (tmp_path / 'model.xyz.original').exists()
    assert not (tmp_path / 'gcmc_subrun.xyz').exists()


def test_gcmc_runs_one_shot_commands_only_in_the_first_sub_run(tmp_path, gpumd_command):
    run_in = (
        'replicate 1 1 2\n'
        'potential nep_C.txt\n'
        'change_box 0.1\n'
        'minimize sd 1.0e-6 10\n'
        'velocity 300 seed 11\n'
        'time_step 1\n'
        'ensemble nvt_lan 300 300 100\n'
        'mc gcmc 10 300 300 1 0.5 1 1 C -100 12.011 auto seed 3\n'
        'dump_thermo 1\n'
        'dump_xyz 10 dump.xyz\n'
        'run 30\n'
    )
    result = _run(tmp_path, gpumd_command, run_in, write_model=_write_cubic_model)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('Start to do an energy minimization.') == 1
    assert result.stdout.count('Change box:') == 1

    model = _make_cubic_model()
    counts = _read_mcmd(tmp_path / 'mcmd.out')[:, 3].astype(int).tolist()
    frames = read(tmp_path / 'dump.xyz', index=':')
    assert [len(frame) for frame in frames] == [2 * len(model)] + counts[:-1]

    expected = model.repeat((1, 1, 2)).cell.array.diagonal() + 0.1
    final = read(tmp_path / 'model.xyz')
    np.testing.assert_allclose(final.cell.array.diagonal(), expected, atol=1e-6)


def test_gcmc_low_chemical_potential_deletes_atoms(tmp_path, gpumd_command):
    result = _run(
        tmp_path,
        gpumd_command,
        _gcmc_run_in(f'mc gcmc 10 300 300 {NUM_SWEEPS} 0.5 1 1 C -100 12.011 auto seed 3'),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    mcmd = _read_mcmd(tmp_path / 'mcmd.out')
    assert mcmd.shape == (3, 6)
    assert mcmd[:, 0].tolist() == [10, 20, 30]
    assert np.all(mcmd[:, 2] == NUM_SWEEPS * NUM_ACTIVE_CELLS)
    counts = mcmd[:, 3].astype(int).tolist()
    assert np.all(mcmd[:, 4] == 0.0)
    assert np.all(np.diff([N_REAL] + counts) <= 0)
    assert counts[-1] < N_REAL

    frames = read(tmp_path / 'dump.xyz', index=':')
    assert [len(frame) for frame in frames] == [N_REAL] + counts[:-1]

    final = read(tmp_path / 'model.xyz')
    assert len(final) == counts[-1]
    assert set(final.get_chemical_symbols()) == {'C'}


def test_gcmc_high_chemical_potential_inserts_atoms(tmp_path, gpumd_command):
    result = _run(
        tmp_path,
        gpumd_command,
        _gcmc_run_in(
            f'mc gcmc 10 300 300 {NUM_SWEEPS} 0.5 2 1 C 100 12.011 auto '
            'parallel seed 5 overlap_cutoff 1 C 1.0'
        ),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    mcmd = _read_mcmd(tmp_path / 'mcmd.out')
    assert mcmd.shape == (3, 7)
    assert mcmd[:, 0].tolist() == [10, 20, 30]
    assert np.all(mcmd[:, 2] == NUM_SWEEPS * NUM_ACTIVE_CELLS)
    counts = mcmd[:, 3].astype(int).tolist()
    assert np.all(mcmd[:, 5] == 0.0)
    assert np.all(np.diff([N_REAL] + counts) >= 0)
    assert counts[-1] > N_REAL

    frames = read(tmp_path / 'dump.xyz', index=':')
    assert [len(frame) for frame in frames] == [N_REAL] + counts[:-1]
    assert len(read(tmp_path / 'model.xyz')) == counts[-1]


@pytest.mark.parametrize(
    'mc_line',
    [
        f'mc gcmc 10 300 300 {NUM_SWEEPS} 0.5 1 1 C -100 12.011 auto seed 3 '
        'overlap_cutoff 1 C 2.0',
        f'mc gcmc 10 300 300 {NUM_SWEEPS} 0.5 2 1 C -100 12.011 auto parallel seed 5 '
        'overlap_cutoff 1 C 2.0',
    ],
)
def test_gcmc_overlap_cutoff_rejects_deletions(tmp_path, gpumd_command, mc_line):
    result = _run(tmp_path, gpumd_command, _gcmc_run_in(mc_line))
    assert result.returncode == 0, result.stdout + result.stderr

    mcmd = _read_mcmd(tmp_path / 'mcmd.out')
    assert mcmd.shape == (3, 7)
    assert np.all(mcmd[:, 3] == N_REAL)
    assert np.all(mcmd[:, 5] == 0.0)
    np.testing.assert_array_equal(mcmd[:, 6], mcmd[:, 2])


def test_gcmc_works_with_other_thermostats(tmp_path, gpumd_command):
    result = _run(
        tmp_path,
        gpumd_command,
        _gcmc_run_in(
            f'mc gcmc 10 300 300 {NUM_SWEEPS} 0.5 1 1 C 100 12.011 auto seed 7',
            ensemble='nvt_ber 300 300 100',
        ),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    thermo = np.loadtxt(tmp_path / 'thermo.out', ndmin=2)
    assert thermo.shape[0] == 30
    assert np.all(np.isfinite(thermo))
    mcmd = _read_mcmd(tmp_path / 'mcmd.out')
    assert mcmd[-1, 3] > N_REAL


@pytest.mark.parametrize(
    'ensemble, pattern, expected',
    [
        (
            'nvt_ber 300 600 100',
            r'initial temperature is (\S+) K\.\n\s*final temperature is (\S+) K\.',
            [(300, 400), (400, 500), (500, 600)],
        ),
        (
            'npt_mttk temp 300 600 iso 0 3 tperiod 100 pperiod 1000',
            r't_start is (\S+), t_stop is (\S+),',
            [(300, 400), (400, 500), (500, 600)],
        ),
        (
            'npt_mttk temp 300 600 iso 0 3 tperiod 100 pperiod 1000',
            r'xx : p_start is (\S+), p_stop is (\S+),',
            [(0, 1), (1, 2), (2, 3)],
        ),
    ],
)
def test_gcmc_splits_the_ensemble_schedule_over_the_whole_run(
    tmp_path, gpumd_command, ensemble, pattern, expected
):
    result = _run(
        tmp_path,
        gpumd_command,
        _gcmc_run_in('mc gcmc 10 300 600 1 0.5 1 1 C -100 12.011 auto seed 3', ensemble=ensemble),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    targets = [tuple(map(float, match)) for match in re.findall(pattern, result.stdout)]
    np.testing.assert_allclose(targets, expected)


def test_gcmc_uses_the_first_potential(tmp_path, gpumd_command):
    e0 = {}
    for first, second in [
        ('nep_C.txt', 'nep_C_shifted.txt'),
        ('nep_C_shifted.txt', 'nep_C.txt'),
    ]:
        run_dir = tmp_path / first.replace('.txt', '')
        run_dir.mkdir()
        _write_potential(run_dir / 'nep_C_shifted.txt', 1000.0)
        run_in = _gcmc_run_in('mc gcmc 10 300 300 1 0.5 1 1 C -100 12.011 auto seed 3').replace(
            'potential nep_C.txt\n', f'potential {first}\npotential {second}\n'
        )
        result = _run(run_dir, gpumd_command, run_in)
        assert result.returncode == 0, result.stdout + result.stderr
        e0[first] = _read_auto_e0(result.stdout)

    assert e0['nep_C_shifted.txt'] - e0['nep_C.txt'] == pytest.approx(1000.0, abs=1e-2)

def test_gcmc_auto_e0_is_the_isolated_atom_energy(tmp_path, gpumd_command):
    result = _run(
        tmp_path,
        gpumd_command,
        'potential nep_C.txt\n'
        'time_step 1\n'
        'ensemble nve\n'
        'mc gcmc 1 300 300 1 0.5 3 1 C 100 12.011 auto seed 3\n'
        'dump_thermo 1\n'
        'run 1\n',
        energy_reference=-5.0,
        write_model=_write_isolated_pair,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    e0 = _read_auto_e0(result.stdout)
    thermo = np.loadtxt(tmp_path / 'thermo.out', ndmin=2)
    assert thermo[0, 2] / 2 == pytest.approx(e0, abs=1e-4)


@pytest.mark.parametrize(
    'mc_line',
    [
        f'mc gcmc 10 300 300 {NUM_SWEEPS} 0.5 1 1 C -100 12.011 auto seed 3',
        f'mc gcmc 10 300 300 {NUM_SWEEPS} 0.5 2 1 C 100 12.011 auto '
        'parallel seed 5 overlap_cutoff 1 C 1.0',
    ],
)
def test_gcmc_auto_e0_cancels_the_energy_reference(tmp_path, gpumd_command, mc_line):
    outputs = {}
    for energy_reference in (0.0, 1000.0):
        run_dir = tmp_path / f'reference_{energy_reference:g}'
        run_dir.mkdir()
        result = _run(
            run_dir, gpumd_command, _gcmc_run_in(mc_line), energy_reference=energy_reference
        )
        assert result.returncode == 0, result.stdout + result.stderr
        outputs[energy_reference] = (
            _read_auto_e0(result.stdout),
            _read_mcmd(run_dir / 'mcmd.out'),
        )

    e0_without, mcmd_without = outputs[0.0]
    e0_with, mcmd_with = outputs[1000.0]
    assert e0_with - e0_without == pytest.approx(1000.0, abs=1e-2)
    np.testing.assert_array_equal(mcmd_with, mcmd_without)
