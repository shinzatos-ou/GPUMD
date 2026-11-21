"""Grand canonical MC (`mc gcmc`): input checks, the sub-run driver, and the chemical-potential
limits of the insertion and deletion acceptance.

The 1024-atom diamond cell (the 16-atom fixture repeated 4 x 4 x 4) is about 16.5 A thick, so it is
not a small box for the 7 A radial cutoff of nep_C.txt. The run of 30 steps is split into three
sub-runs of 10 steps, each followed by one call of the trials.
"""
import shutil
import subprocess

import numpy as np
import pytest
from ase.io import read

from conftest import MODELS_DIR, make_bulk_C

N_REAL = 1024
NUM_SWEEPS = 20
NUM_ACTIVE_CELLS = 1


def _write_model(path):
    atoms = make_bulk_C().repeat(4)
    assert len(atoms) == N_REAL
    cell = atoms.cell.array
    lattice = ' '.join(f'{x:.10f}' for x in cell.flatten())
    lines = [
        f'{len(atoms)}',
        f'Lattice="{lattice}" pbc="T T T" Properties=species:S:1:pos:R:3',
    ]
    for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions):
        lines.append(f'{symbol} {position[0]:.10f} {position[1]:.10f} {position[2]:.10f}')
    path.write_text('\n'.join(lines) + '\n')


def _run(tmp_path, gpumd_command, run_in):
    shutil.copy(MODELS_DIR / 'nep_C.txt', tmp_path / 'nep_C.txt')
    _write_model(tmp_path / 'model.xyz')
    (tmp_path / 'run.in').write_text(run_in)
    return subprocess.run([gpumd_command], cwd=tmp_path, capture_output=True, text=True)


def _read_mcmd(path):
    return np.loadtxt(path, ndmin=2)


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
            'deposit 10 2 5.0 atom C 1 0\n'
            'run 10\n',
            'cannot be used together with deposit',
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
            f'mc gcmc 10 300 300 {NUM_SWEEPS} 0.5 1 1 C 100 12.011 auto '
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
