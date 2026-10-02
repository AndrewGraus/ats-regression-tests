#!/usr/bin/env python3
"""Check that EcoSIM's surface water source actually reaches ATS overland flow.

With "water source in meters" = true, overland flow uses
surface-ecosim_water_source_mols = surface-ecosim_water_source *
surface-source_molar_density. This checker reads the surface visualization of
a run and requires that product at every visualization cycle. In particular it
fails if the source EcoSIM wrote is non-zero while the field ATS uses is zero.

Usage: check_ecosim_coupling.py <run_directory> [<run_directory> ...]
"""

from pathlib import Path
import sys

try:
    import h5py
except ImportError as error:
    raise SystemExit("This checker requires the Python package h5py.") from error


SOURCE = "surface-ecosim_water_source"
SOURCE_MOLS = "surface-ecosim_water_source_mols"
MOLAR_DENSITY = "surface-source_molar_density"
RELATIVE_TOLERANCE = 1.0e-12


def check_run(run_directory):
    filename = Path(run_directory) / "ats_vis_surface_data.h5"
    if not filename.is_file():
        raise FileNotFoundError(f"required file does not exist: {filename}")

    with h5py.File(filename, "r") as fid:
        for field in (SOURCE, SOURCE_MOLS, MOLAR_DENSITY):
            if field not in fid:
                raise KeyError(f"group {field!r} is missing from {filename}")
        cycles = sorted(fid[SOURCE].keys(), key=int)
        nonzero, bad = 0, []
        for cycle in cycles:
            # dataset values only; each dataset also carries a "Time" attribute
            src = fid[SOURCE][cycle][:].ravel()
            mols = fid[SOURCE_MOLS][cycle][:].ravel()
            dens = fid[MOLAR_DENSITY][cycle][:].ravel()
            for s, m, d in zip(src, mols, dens):
                expected = s * d
                if s != 0.0:
                    nonzero += 1
                scale = max(abs(expected), abs(m))
                if scale > 0.0 and abs(m - expected) > RELATIVE_TOLERANCE * scale:
                    bad.append((int(cycle), s, m, expected))

    print(f"{run_directory}: {len(cycles)} cycles, EcoSIM source non-zero in {nonzero} cell-cycles, "
          f"{len(bad)} mismatches")
    for cycle, s, m, expected in bad[:5]:
        print(f"   cycle {cycle}: source {s:.6g} m/s, ATS uses {m:.6g} mol/m2/s, expected {expected:.6g}")
    return not bad


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    ok = all([check_run(run) for run in sys.argv[1:]])
    if not ok:
        print("FAIL: the water source ATS uses does not match the one EcoSIM wrote")
        return 1
    print("PASS: EcoSIM's surface water source reaches ATS overland flow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
