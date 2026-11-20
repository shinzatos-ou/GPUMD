"""Grand canonical MC (`mc gcmc`): input checks, ghost-atom handling, and the chemical-potential
limits of the insertion and deletion acceptance.

The 1024-atom diamond cell (the 16-atom fixture repeated 4 x 4 x 4) is about 16.5 A thick, so it is
not a small box for the 7 A radial cutoff of nep_C.txt. Ghost slots are reserved as many as the
real atoms, so that counting them as atoms would halve the reported temperature.
"""
import shutil
import subprocess

import numpy as np
import pytest
from ase.io import read

from conftest import MODELS_DIR, make_bulk_C

N_REAL = 1024
MAX_GHOST = 1024
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


def _gcmc_run_in(mc_line):
    return (
        f'{mc_line}\n'
        'potential nep_C.txt\n'
        'velocity 300 seed 11\n'
        'time_step 1\n'
        'ensemble nvt_lan 300 300 100\n'
        'dump_thermo 1\n'
        'dump_xyz 10 dump.xyz\n'
        'dump_restart 30\n'
        'run 30\n'
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    'run_in, message',
    [
        (
            'potential nep_C.txt\n'
            f'mc gcmc 10 300 300 1 0.5 1 {MAX_GHOST} 1 C -7 12.011 auto\n'
            'run 1\n',
            'mc gcmc must appear before potential',
        ),
        (
            f'mc gcmc 10 300 300 1 1.0 1 {MAX_GHOST} 1 C -7 12.011 auto\n'
            'potential nep_C.txt\n'
            'run 1\n',
            'a_plus should be in (0, 1)',
        ),
        (
            f'mc gcmc 10 300 300 1 0.5 1 {MAX_GHOST} 1 H -7 1.008 auto\n'
            'potential nep_C.txt\n'
            'run 1\n',
            'The active species H is not supported by the potential.',
        ),
        (
            f'mc gcmc 10 300 300 1 0.5 1 {MAX_GHOST} 1 C -7 12.011 auto unknown\n'
            'potential nep_C.txt\n'
            'run 1\n',
            'unknown option for mc gcmc',
        ),
        (
            f'mc gcmc 10 300 300 1 0.5 1 {MAX_GHOST} 1 C -7 12.011 auto overlap_cutoff 1 C 8.0\n'
            'potential nep_C.txt\n'
            'run 1\n',
            'overlap_cutoff for mc gcmc should not exceed the NEP radial cutoff',
        ),
    ],
)
def test_gcmc_rejects_invalid_input(tmp_path, gpumd_command, run_in, message):
    result = _run(tmp_path, gpumd_command, run_in)
    assert result.returncode != 0, result.stdout + result.stderr
    assert message in result.stdout + result.stderr


def test_gcmc_ghost_atoms_are_not_counted_in_temperature(tmp_path, gpumd_command):
    result = _run(
        tmp_path,
        gpumd_command,
        _gcmc_run_in(f'mc gcmc 10 300 300 1 0.5 1 {MAX_GHOST} 1 C -100 12.011 auto seed 3'),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    thermo = np.loadtxt(tmp_path / 'thermo.out', ndmin=2)
    assert np.all(np.isfinite(thermo))
    assert 270.0 < thermo[0, 0] < 330.0


def test_gcmc_low_chemical_potential_deletes_atoms(tmp_path, gpumd_command):
    result = _run(
        tmp_path,
        gpumd_command,
        _gcmc_run_in(
            f'mc gcmc 10 300 300 {NUM_SWEEPS} 0.5 1 {MAX_GHOST} 1 C -100 12.011 auto seed 3'
        ),
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
    assert [len(frame) for frame in frames] == counts
    for frame in frames:
        assert set(frame.get_chemical_symbols()) == {'C'}

    restart = read(tmp_path / 'restart.xyz')
    assert len(restart) == counts[-1]
    assert set(restart.get_chemical_symbols()) == {'C'}


def test_gcmc_high_chemical_potential_inserts_atoms(tmp_path, gpumd_command):
    result = _run(
        tmp_path,
        gpumd_command,
        _gcmc_run_in(
            f'mc gcmc 10 300 300 {NUM_SWEEPS} 0.5 1 {MAX_GHOST} 1 C 100 12.011 auto '
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
    assert [len(frame) for frame in frames] == counts
    assert len(read(tmp_path / 'restart.xyz')) == counts[-1]
