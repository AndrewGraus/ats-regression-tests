#!/usr/bin/env python3
"""Check that per-area results do not depend on the column footprint.

Each pair runs the same case on a 1 m x 1 m and a 2 m x 2 m column:
  short (default): ecosim_column_area_{1m2,4m2}, the 1-day freeze-thaw case
  long:            ecosim_column_area_long_{1m2,4m2}, the 200-day case with snowmelt
Usage: check_column_area.py [short|long]. All forcing and parameters are per unit area,
so per-area quantities must agree. Per-cell totals (water content, volumes,
face fluxes, EcoSIM's "per grid cell" snow and litter volumes) scale with the
area and are not compared.

A missing division by the column area in a coupled flux shows up as a ratio of
4 between the two runs (e.g. surface-ecosim_water_source).
"""

from pathlib import Path
import sys

try:
    import h5py
except ImportError as error:
    raise SystemExit("This checker requires the Python package h5py.") from error


HERE = Path(__file__).resolve().parent
PAIRS = {
    "short": "ecosim_column_area",
    "long": "ecosim_column_area_long",
}
RELATIVE_TOLERANCE = 1.0e-10

CHECKPOINT_FIELDS = [
    "pressure.cell.0",
    "temperature.cell.0",
    "surface-pressure.cell.0",
    "surface-temperature.cell.0",
    "surface-snow_depth.cell.0",
] + [
    f"surface-snow_layer_{name}.cell.{layer}"
    for name in ("temperature", "temperature_c", "density", "thickness")
    for layer in range(5)
]

# compared at every visualization cycle: file -> fields
VIS_FIELDS = {
    "ats_vis_surface_data.h5": (
        "surface-ecosim_water_source",
        "surface-ecosim_source",
        "surface-pressure",
        "surface-snow_depth",
    ),
    # subsurface visualization, written only by the long pair
    "ats_vis_data.h5": (
        "pressure",
        "temperature",
        "saturation_liquid",
    ),
}


def require_file(filename):
    if not filename.is_file():
        raise FileNotFoundError(f"required file does not exist: {filename}")


def max_relative_difference(a, b):
    worst = 0.0
    for x, y in zip(a, b):
        scale = max(abs(x), abs(y))
        if scale > 0.0:
            worst = max(worst, abs(x - y) / scale)
    return worst


def read_checkpoint(run_directory):
    filename = run_directory / "checkpoint_final.h5"
    require_file(filename)
    values = {}
    with h5py.File(filename, "r") as fid:
        for field in CHECKPOINT_FIELDS:
            if field not in fid:
                raise KeyError(f"dataset {field!r} is missing from {filename}")
            values[field] = [float(v) for v in fid[field][:].ravel()]
    return values


def read_vis(run_directory):
    """Dataset values only; each dataset also carries a "Time" attribute."""
    values = {}
    for vis_file, fields in VIS_FIELDS.items():
        filename = run_directory / vis_file
        if not filename.is_file():
            continue
        with h5py.File(filename, "r") as fid:
            for field in fields:
                if field not in fid:
                    raise KeyError(f"group {field!r} is missing from {filename}")
                cycles = sorted(fid[field].keys(), key=int)
                values[field] = {c: [float(v) for v in fid[field][c][:].ravel()] for c in cycles}
    return values


def main():
    pair = sys.argv[1] if len(sys.argv) > 1 else "short"
    if pair not in PAIRS:
        raise SystemExit(f"unknown pair {pair!r}; use one of {', '.join(PAIRS)}")
    RUNS = {area: HERE / f"{PAIRS[pair]}_{area}" for area in ("1m2", "4m2")}
    failures = []

    checkpoints = {name: read_checkpoint(path) for name, path in RUNS.items()}
    print(f"{'checkpoint field':42s} {'max rel. difference':>20s}")
    for field in CHECKPOINT_FIELDS:
        diff = max_relative_difference(checkpoints["1m2"][field], checkpoints["4m2"][field])
        print(f"{field:42s} {diff:20.3e}")
        if diff > RELATIVE_TOLERANCE:
            failures.append(field)

    vis = {name: read_vis(path) for name, path in RUNS.items()}
    print(f"\n{'vis field':42s} {'max rel. difference':>20s} {'max ratio 4m2/1m2':>18s}")
    for field in vis["1m2"]:
        if field not in vis["4m2"]:
            failures.append(f"{field} (missing in 4m2 run)")
            continue
        a, b = vis["1m2"][field], vis["4m2"][field]
        if set(a) != set(b):
            failures.append(f"{field} (different cycles)")
            continue
        diff, ratio = 0.0, 1.0
        for cycle in a:
            diff = max(diff, max_relative_difference(a[cycle], b[cycle]))
            for x, y in zip(a[cycle], b[cycle]):
                if x != 0.0 and abs(y / x - 1.0) > abs(ratio - 1.0):
                    ratio = y / x
        print(f"{field:42s} {diff:20.3e} {ratio:18.6g}")
        if diff > RELATIVE_TOLERANCE:
            failures.append(field)

    if failures:
        print(f"\nFAIL: results depend on the column area: {', '.join(failures)}")
        return 1
    print("\nPASS: per-area results are independent of the column area")
    return 0


if __name__ == "__main__":
    sys.exit(main())
