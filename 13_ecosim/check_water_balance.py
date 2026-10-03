#!/usr/bin/env python3
"""ATS water balance and EcoSIM coupling check for runs coupled through the
surface water source.

Reads water_balance_surface.dat and water_balance_subsurface.dat (written every
cycle by the "observations" block of the 13_ecosim inputs) from each run
directory and reports:

R1, ATS conservation. The columns have no flow boundary conditions, so
    change in (subsurface + surface) water content = water applied by
    overland flow through surface-ecosim_water_source_mols, plus, if observed,
    the subsurface source subsurface_ecosim_water_source.
    EcoSIM runs last in the MPC, so the rate observed at the end of cycle k-1
    is the one applied during cycle k. R1 is relative to the largest of
    |storage change|, total |applied| and |reported| water, or passes if
    its residual is below ABSOLUTE_TOLERANCE_MM.

R2, coupling. Water ATS applied = water EcoSIM reported. EcoSIM reports a
    rate for the hour since its previous call. Calls are reproduced from the
    EcoSIM PK's gate (EcoSIM::AdvanceSingleProcess): EcoSIM runs at the end of
    a step ending at t if t - t_last >= 3600 s, with t_last starting at 0.
    Update this rule if the coupling clock changes. The value reported at the
    final cycle is applied only after the run ends, so it is excluded and
    printed as "pending".

Usage: check_water_balance.py <run_dir> [<run_dir> ...]
Exit status 1 if any run fails a check.
"""

from pathlib import Path
import sys

RELATIVE_TOLERANCE = 1.0e-6
# R1 also passes if its residual is below this depth of water; it covers periods
# with no water exchange, where the relative R1 is solver drift divided by itself
# (observed drift: up to 2e-6 mm over 200 days)
ABSOLUTE_TOLERANCE_MM = 1.0e-5
ECOSIM_PERIOD = 3600.0  # s, the gate in EcoSIM::AdvanceSingleProcess

SURFACE_FILE = "water_balance_surface.dat"
SUBSURFACE_FILE = "water_balance_subsurface.dat"
COLUMNS = {
    "surface water content [mol]": "w_surf",
    "ecosim water source [m^3 s^-1]": "src",
    "applied water source [mol s^-1]": "mols",
    "source molar density [mol m^-3]": "dens",
    "surface area [m^2]": "area",
    "subsurface water content [mol]": "w_sub",
    # optional: inputs with a subsurface EcoSIM source (prescribed phenology)
    "subsurface ecosim water source [integral]": "src_sub",
}


def read_observations(filename):
    """Return {short name: [values]} including "time" from an ATS observation file."""
    if not filename.is_file():
        raise FileNotFoundError(f"required file does not exist: {filename}")
    names, rows = None, []
    for line in filename.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith('"'):
            # column header: "time [s]","<observation name>",...
            names = []
            for name in (n.strip().strip('"') for n in line.split(",")):
                if name == "time [s]":
                    names.append("time")
                elif name in COLUMNS:
                    names.append(COLUMNS[name])
                else:
                    raise KeyError(f"unexpected observation {name!r} in {filename}")
            continue
        rows.append([float(v) for v in line.split(",")])
    if names is None:
        raise ValueError(f"no column header in {filename}")
    data = {name: [row[i] for row in rows] for i, name in enumerate(names)}
    return data


def check_run(run_directory):
    run = Path(run_directory)
    surf = read_observations(run / SURFACE_FILE)
    sub = read_observations(run / SUBSURFACE_FILE)
    if surf["time"] != sub["time"]:
        raise ValueError(f"{run}: surface and subsurface observation times differ")

    t = surf["time"]
    n = len(t)
    if n < 2:
        raise ValueError(f"{run}: need at least two observation cycles")
    storage = [a + b for a, b in zip(surf["w_surf"], sub["w_sub"])]
    dens, area = surf["dens"][-1], surf["area"][-1]

    # R1: applied water, rate from the previous cycle (lagged), and unlagged as a diagnostic.
    # The subsurface source is applied by ATS as mol m^-3 s^-1, so its integral is mol s^-1.
    sub_rate = sub.get("src_sub", [0.0] * n)
    rate = [m + q for m, q in zip(surf["mols"], sub_rate)]
    applied_sub = sum((t[k] - t[k - 1]) * sub_rate[k - 1] for k in range(1, n))
    applied_total = sum((t[k] - t[k - 1]) * rate[k - 1] for k in range(1, n))
    applied_unlagged = sum((t[k] - t[k - 1]) * rate[k] for k in range(1, n))
    applied_abs = sum(abs((t[k] - t[k - 1]) * rate[k - 1]) for k in range(1, n))
    applied = applied_total - applied_sub  # surface source, for R2
    d_storage = storage[-1] - storage[0]

    # R2: water EcoSIM reported at each gated call, excluding the final (pending) cycle
    t_last, reported, pending, calls = 0.0, 0.0, 0.0, 0
    for k in range(1, n):
        if t[k] - t_last >= ECOSIM_PERIOD:
            water = surf["src"][k] * ECOSIM_PERIOD * surf["dens"][k]
            if k == n - 1:
                pending = water
            else:
                reported += water
            calls += 1
            t_last = t[k]
    r2 = (applied - reported) / max(abs(applied), abs(reported), 1.0e-300)

    # R1 is scaled by the water passing through the column, including what EcoSIM
    # reported, so that solver-level drift is not divided by itself when nothing was applied
    scale = max(abs(d_storage), applied_abs, abs(reported), 1.0e-300)
    r1 = (d_storage - applied_total) / scale
    r1_unlagged = (d_storage - applied_unlagged) / scale

    to_mm = 1.0e3 / (dens * area)
    print(f"{run}: {n - 1} steps, {calls} EcoSIM calls, t = {t[0]:.6g} .. {t[-1]:.6g} s, area {area:g} m2")
    print(f"   storage change        {d_storage:16.8e} mol  {d_storage * to_mm:12.5g} mm")
    print(f"   applied by ATS        {applied:16.8e} mol  {applied * to_mm:12.5g} mm")
    if "src_sub" in sub:
        print(f"   applied, subsurface   {applied_sub:16.8e} mol  {applied_sub * to_mm:12.5g} mm"
              "   (not in R2)")
    print(f"   reported by EcoSIM    {reported:16.8e} mol  {reported * to_mm:12.5g} mm"
          f"   (pending at end: {pending * to_mm:.5g} mm)")
    print(f"   R1 ATS conservation   {r1:12.3e}   (residual {d_storage - applied_total:.3e} mol, "
          f"without lag: {r1_unlagged:.3e})")
    print(f"   R2 coupling           {r2:12.3e}")
    r1_ok = abs(r1) <= RELATIVE_TOLERANCE or abs((d_storage - applied_total) * to_mm) <= ABSOLUTE_TOLERANCE_MM
    if r1_ok and abs(r1) > RELATIVE_TOLERANCE:
        print(f"   R1 passes on the absolute tolerance ({ABSOLUTE_TOLERANCE_MM:g} mm): "
              f"residual {(d_storage - applied_total) * to_mm:.3g} mm")
    ok = r1_ok and abs(r2) <= RELATIVE_TOLERANCE
    print(f"   {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    results = [check_run(run) for run in sys.argv[1:]]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
