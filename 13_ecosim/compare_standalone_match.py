#!/usr/bin/env python3
"""Compare the coupled ATS-EcoSIM column with the matched standalone EcoSIM run.

Usage: compare_standalone_match.py <ats_run_dir> <ecosim_run_dir> [--days D [D ...]]

ats_run_dir:    run of ecosim_prescribed_phenology_standalone_match.xml
ecosim_run_dir: run of data/standalone_match/standalone_match.namelist

Writes comparison.md and comparison_*.png in ecosim_run_dir. Fluxes are in
mm (1 m2 column): negative = water leaving the soil or the surface.

Figures:
  comparison_fluxes.png      daily transpiration, root uptake, ground + snow evaporation, snow depth
  comparison_profiles.png    theta and temperature profiles at the --days
  comparison_saturation.png  liquid and liquid + ice saturation along the column at the --days
                             (one panel per day, depth on a log scale)
  comparison_temperature.png soil temperature along the column at the --days (same layout)
  comparison_snow.png        snow depth through the year, --days marked

Times: ATS visualization is daily at t = d days; EcoSIM history record k is the
end of hour k+1, so t = d days is record 24 d - 1.

Quantities, coupled (ATS run) against standalone (EcoSIM history):
  transpiration        surface-transpiration (EcoSIM a_Transpiration, m3 per
                       grid cell per hour)            vs QTransp_col [mm h-1]
  root uptake          subsurface_ecosim_water_source applied by ATS [mol s-1]
                       vs RootH2OUptk_pvr summed over layers and PFTs [mm h-1]
                       (RootH2OUptake_vr is not filled in prescribed phenology)
  ground evaporation   surface-evaporation_ground + sublimation_snow + evaporation_snow
                       (EcoSIM, m3 per grid cell per hour) vs EVAPGrnd_col [mm h-1].
                       Standalone EVAPGrnd_col is VapXAir2GSurf_col = soil + snow vapor
                       exchange; the coupled evaporation_ground is the soil part only
                       (TEvapXAir2Toplay_col), so the snow part is added to compare
  canopy evaporation   evaporation_canopy (same units)  vs QVegEvap_col [mm h-1]
  soil water storage   subsurface water content [mol]   vs sum of (liquid + 0.917 ice) * pore * dz
  snow depth           surface-snow_depth [m] (daily)  vs SNOWPACK_col/SNOWDENS_col
  soil water, temperature by layer: ATS vis (daily)    vs (rWatFLP_vr + rIceFLP_vr)*VSPore_vr, TEMP_vr
                       (theta = liquid + ice volume in both)
"""

import argparse
import datetime
import glob
import os

import h5py
import netCDF4
import numpy as np

MOLAR_DENSITY = 1000.0 / 0.0180153     # mol m-3, liquid water (ATS EOS, 1000 kg m-3)
ICE_DENSITY = 0.917                    # EcoSIM DENSICE, ice volume -> water volume
AREA = 1.0                             # m2


def read_dat(path):
    names, rows = [], []
    for line in open(path):
        if line.startswith("# Observation Name:"):
            names.append(line.split(":", 1)[1].strip())
        elif not line.startswith("#") and not line.startswith('"') and line.strip():
            rows.append([float(v) for v in line.split(",")])
    a = np.array(rows)
    return {n: a[:, i] for i, n in enumerate(names)}


def column(d, prefix):
    keys = [k for k in d if k.startswith(prefix)]
    if len(keys) != 1:
        raise KeyError(f"{prefix!r}: {keys}")
    return d[keys[0]]


def ats_hourly(run):
    """per EcoSIM call (hour): time [s] and fluxes [mm h-1]"""
    e = read_dat(os.path.join(run, "ecosim_water_outputs.dat"))
    b = read_dat(os.path.join(run, "water_balance_subsurface.dat"))
    t = e["time [s]"]
    calls, t_last = [], 0.0
    for k in range(1, len(t)):                        # EcoSIM's 3600 s gate (as check_water_balance)
        if t[k] - t_last >= 3600.0:
            calls.append(k)
            t_last = t[k]
    calls = np.array(calls)
    to_mm = 1000.0 / AREA                             # m3 per grid cell -> mm
    return {
        "time": t[calls],
        "transpiration": column(e, "transpiration")[calls] * to_mm,
        # soil + snow vapor exchange, the quantity standalone EVAPGrnd_col reports
        "evaporation_ground": (column(e, "evaporation_ground")[calls] + column(e, "sublimation_snow")[calls]
                               + column(e, "evaporation_snow")[calls]) * to_mm,
        "evaporation_canopy": column(e, "evaporation_canopy")[calls] * to_mm,
        # change of soil water since the start [mm]
        "storage": (column(b, "subsurface water content")[calls]
                    - column(b, "subsurface water content")[0]) / MOLAR_DENSITY * to_mm,
        # applied rate [mol s-1] over the next hour -> mm h-1
        "root_uptake": column(b, "subsurface ecosim water source [integral]")[calls]
                       * 3600.0 / MOLAR_DENSITY * to_mm,
    }


def ats_profiles(run):
    """daily cell profiles from the visualization files, ordered top -> bottom"""
    with h5py.File(os.path.join(run, "ats_vis_mesh.h5"), "r") as m:
        mesh = m[list(m.keys())[0]]["Mesh"]
        nodes = mesh["Nodes"][:]
        mixed = mesh["MixedElements"][:].ravel()
    centroids, i = [], 0
    while i < len(mixed):                              # hexes: type id + 8 nodes
        conn = mixed[i + 1:i + 9]
        centroids.append(nodes[conn, 2].mean())
        i += 9
    order = np.argsort(-np.array(centroids))
    out = {"z": np.array(centroids)[order], "top": nodes[:, 2].max()}
    out["depth"] = out["top"] - out["z"]               # cell centre depth below the surface [m]
    with h5py.File(os.path.join(run, "ats_vis_data.h5"), "r") as f:
        def series(prefix):
            key = [k for k in f if k.startswith(prefix)][0]
            cycles = sorted(f[key].keys(), key=int)
            return np.array([f[key][c][:, 0][order] for c in cycles])
        cycles = sorted(f[[k for k in f if k.startswith("saturation_liquid")][0]].keys(), key=int)
        out["time"] = np.array([f[[k for k in f if k.startswith("saturation_liquid")][0]][c]
                                .attrs.get("Time", np.nan) for c in cycles])
        sat = series("saturation_liquid")
        ice = series("saturation_ice")
        por = series("porosity")
        out["sat_liquid"] = sat
        out["sat_total"] = sat + ice
        out["theta"] = (sat + ice) * por
        out["temperature"] = series("temperature") - 273.15
    return out


def ats_snow_depth(run):
    d = read_dat(os.path.join(run, "surf_prop.dat"))
    return d["time [s]"], column(d, "snow_depth")


def ecosim_history(run):
    files = sorted(glob.glob(os.path.join(run, "*.ecosim.h0.*.nc")))
    if not files:
        raise FileNotFoundError(f"no EcoSIM history in {run}")
    data = {}
    for f in files:
        with netCDF4.Dataset(f) as d:
            for k in ["QTransp_col", "EVAPGrnd_col", "QVegEvap_col", "SNOWPACK_col", "SNOWDENS_col",
                      "ECO_LAI_col", "PREC_col"]:
                data.setdefault(k, []).append(np.ma.filled(d[k][:, 0], np.nan))
            for k in ["rWatFLP_vr", "rIceFLP_vr", "VSPore_vr", "TEMP_vr", "BOTDEPZ_vr"]:
                data.setdefault(k, []).append(np.ma.filled(d[k][:, :, 0], np.nan))
            data.setdefault("RootH2OUptk", []).append(
                np.ma.filled(d["RootH2OUptk_pvr"][:].astype(float), 0.0).sum(axis=(1, 2)))
    return {k: np.concatenate(v) for k, v in data.items()}


def daily(x, per_day=24):
    n = len(x) // per_day
    return x[:n * per_day].reshape(n, per_day).sum(axis=1)


def eco_index(day, nrec):
    """EcoSIM history record at t = day days (record k = end of hour k+1)"""
    return int(min(max(24 * day - 1, 0), nrec - 1))


YEAR = 1996   # set in main from the EcoSIM history file name


def day_label(day, year=None):
    year = YEAR if year is None else year
    date = datetime.date(year, 1, 1) + datetime.timedelta(days=int(day))
    return f"day {day} ({date.strftime('%d %b')})"


def depth_axis(ax, prof):
    """log depth axis, surface at the top, plain tick labels"""
    ax.set_yscale("log")
    ticks = [x for x in (0.03, 0.1, 0.3, 1, 3, 10, 25) if prof["depth"].min() * 0.8 <= x <= prof["top"]]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{x:g}" for x in ticks])
    ax.minorticks_off()
    ax.set_ylim(prof["depth"].max() * 1.2, prof["depth"].min() * 0.8)
    ax.set_ylabel("depth of layer centre [m] (log)")


def plot_temperature(plt, prof, e, layers, mid_e, days, path):
    """soil temperature profiles, one panel per day (same layout as plot_saturation)"""
    temp_e = e["TEMP_vr"][:, layers]
    fig, ax = plt.subplots(1, len(days), figsize=(3.0 * len(days), 6.5), sharey=True, squeeze=False)
    ax = ax[0]
    for axi, d in zip(ax, days):
        h = eco_index(d, len(temp_e))
        axi.axvline(0.0, color="0.6", ls=":", lw=1)
        axi.plot(prof["temperature"][d], prof["depth"], "-", color="C0", label="ATS")
        axi.plot(temp_e[h], mid_e, "--", color="C1", label="EcoSIM")
        axi.set_title(day_label(d), fontsize=10)
        axi.set_xlabel("temperature [C]")
        axi.grid(alpha=0.3)
    depth_axis(ax[0], prof)
    ax[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Soil temperature along the column: coupled ATS-EcoSIM (solid) vs standalone EcoSIM "
                 "(dashed); dotted: 0 C", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plot_saturation(plt, prof, e, layers, mid_e, days, path):
    """liquid and liquid + ice saturation profiles, one panel per day"""
    sat_liq_e = e["rWatFLP_vr"][:, layers]
    sat_tot_e = (e["rWatFLP_vr"] + e["rIceFLP_vr"])[:, layers]
    fig, ax = plt.subplots(1, len(days), figsize=(3.0 * len(days), 6.5), sharey=True, squeeze=False)
    ax = ax[0]
    for axi, d in zip(ax, days):
        h = eco_index(d, len(sat_liq_e))
        axi.plot(prof["sat_total"][d], prof["depth"], "-", color="C0", lw=2.5, alpha=0.4,
                 label="ATS liquid + ice")
        axi.plot(prof["sat_liquid"][d], prof["depth"], "-", color="C0", label="ATS liquid")
        axi.plot(sat_tot_e[h], mid_e, "--", color="C1", lw=2.5, alpha=0.4, label="EcoSIM liquid + ice")
        axi.plot(sat_liq_e[h], mid_e, "--", color="C1", label="EcoSIM liquid")
        axi.set_title(day_label(d), fontsize=10)
        axi.set_xlim(0, 1.05)
        axi.set_xlabel("saturation [-]")
        axi.grid(alpha=0.3)
    depth_axis(ax[0], prof)
    ax[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Saturation along the column: coupled ATS-EcoSIM (solid) vs standalone EcoSIM (dashed)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plot_snow(plt, ts, snow_ats, snow_e, days, path):
    """snow depth through the year, with the profile days marked"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts / 86400.0, snow_ats, label="coupled ATS-EcoSIM")
    ax.plot((np.arange(len(snow_e)) + 1) / 24.0, np.nan_to_num(snow_e), label="standalone EcoSIM")
    for d in days:
        ax.axvline(d, color="0.6", ls=":", lw=1)
        ax.text(d, ax.get_ylim()[1] * 0.97, f" {d}", fontsize=7, color="0.4", va="top")
    ax.set_xlabel(f"day of {YEAR}")
    ax.set_ylabel("snow depth [m]")
    ax.set_title("Snow depth (dotted lines: days of the saturation profiles)", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ats_run")
    parser.add_argument("eco_run")
    parser.add_argument("--days", type=int, nargs="+", default=[30, 75, 120, 200, 300],
                        help="days of the year for the profile figures and table (default: winter, late "
                             "winter, melt, summer, autumn)")
    args = parser.parse_args()
    ats_run, eco_run = args.ats_run, args.eco_run
    global YEAR
    hist = sorted(glob.glob(os.path.join(eco_run, "*.ecosim.h0.*.nc")))
    if hist:
        YEAR = int(os.path.basename(hist[0]).split(".h0.")[1][:4])
    a = ats_hourly(ats_run)
    prof = ats_profiles(ats_run)
    ts, snow_ats = ats_snow_depth(ats_run)
    e = ecosim_history(eco_run)

    # EcoSIM layers present in the history (non-zero bottom depth), top -> bottom
    bot = e["BOTDEPZ_vr"][0]
    layers = np.where(bot > 0)[0]
    nl = min(len(layers), prof["theta"].shape[1])
    layers = layers[:nl]
    mid_e = bot[layers] - 0.5 * np.diff(np.concatenate([[0.0], bot[layers]]))   # EcoSIM layer centres

    nh = min(len(a["time"]), len(e["QTransp_col"]))
    dz = np.diff(np.concatenate([[0.0], bot[layers]]))
    storage_e = (((e["rWatFLP_vr"][:, layers] + ICE_DENSITY * e["rIceFLP_vr"][:, layers])
                  * e["VSPore_vr"][:, layers]) * dz).sum(axis=1) * 1000.0
    rows = []
    def total(name, ats, eco):
        rows.append((name, ats[:nh].sum(), eco[:nh].sum()))
    rows.append(("precipitation (same forcing)", np.nan, e["PREC_col"][:nh].sum()))
    total("transpiration", a["transpiration"], e["QTransp_col"])
    total("root uptake (soil -> plant)", a["root_uptake"], e["RootH2OUptk"])
    total("ground + snow evaporation/sublimation", a["evaporation_ground"], e["EVAPGrnd_col"])
    total("canopy evaporation", a["evaporation_canopy"], e["QVegEvap_col"])
    # storage: change from the initial state (ATS) and from the end of hour 1 (EcoSIM history)
    rows.append(("soil water storage change", a["storage"][nh - 1], storage_e[nh - 1] - storage_e[0]))

    lines = ["# Coupled ATS-EcoSIM vs standalone EcoSIM (matched column)", "",
             f"ATS run: `{ats_run}`  EcoSIM run: `{eco_run}`  ({nh} hours compared)", "",
             f"Canopy LAI: standalone history {np.nanmean(e['ECO_LAI_col']):.4f} "
             f"(min {np.nanmin(e['ECO_LAI_col']):.4f}, max {np.nanmax(e['ECO_LAI_col']):.4f})", "",
             "## Totals over the compared period [mm] (negative = leaving the soil/surface)", "",
             "| quantity | coupled (ATS) | standalone EcoSIM | difference |", "|---|---|---|---|"]
    for name, x, y in rows:
        lines.append(f"| {name} | {x:.3f} | {y:.3f} | {x - y:.3f} |" if np.isfinite(x)
                     else f"| {name} | | {y:.3f} | |")

    lines += ["", "## Layer bottom depths [m]: ATS (from cell centres) vs EcoSIM", "",
              "| layer | ATS | EcoSIM |", "|---|---|---|"]
    b = 0.0
    for i in range(nl):
        b = 2.0 * prof["depth"][i] - b               # centre is midway between the faces
        lines.append(f"| {i+1} | {b:.4f} | {bot[layers[i]]:.4f} |")

    # soil water and temperature at selected days and depths
    days = [d for d in args.days if 0 <= d < len(prof["time"])]
    sel = [0, nl // 4, nl // 2, nl - 1]
    lines += ["", "## Soil water theta [m3 m-3] and temperature [C], coupled / standalone", "",
              "| day | layer | depth [m] | theta ATS | theta EcoSIM | T ATS | T EcoSIM |",
              "|---|---|---|---|---|---|---|"]
    theta_e = (e["rWatFLP_vr"] + e["rIceFLP_vr"]) * e["VSPore_vr"]
    for d in days:
        h = eco_index(d, len(theta_e))
        for i in sel:
            lines.append(f"| {d} | {i+1} | {prof['depth'][i]:.3f} | {prof['theta'][d, i]:.4f} | "
                         f"{theta_e[h, layers[i]]:.4f} | {prof['temperature'][d, i]:.2f} | "
                         f"{e['TEMP_vr'][h, layers[i]]:.2f} |")

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
        dd = np.arange(nh // 24)
        for axi, (name, x, y) in zip(ax[:3], [
                ("transpiration", a["transpiration"], e["QTransp_col"]),
                ("root uptake", a["root_uptake"], e["RootH2OUptk"]),
                ("ground + snow\nevaporation", a["evaporation_ground"], e["EVAPGrnd_col"])]):
            axi.plot(dd, daily(x[:nh]), label="coupled ATS-EcoSIM")
            axi.plot(dd, daily(y[:nh]), label="standalone EcoSIM")
            axi.set_ylabel(f"{name}\n[mm d-1]")
        snow_e = e["SNOWPACK_col"] / np.where(e["SNOWDENS_col"] > 0, e["SNOWDENS_col"], np.nan)
        ax[3].plot(ts / 86400.0, snow_ats, label="coupled ATS-EcoSIM")
        ax[3].plot((np.arange(len(snow_e)) + 1) / 24.0, np.nan_to_num(snow_e), label="standalone EcoSIM")
        ax[3].set_ylabel("snow depth [m]")
        ax[3].set_xlabel(f"day of {YEAR}")
        ax[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(eco_run, "comparison_fluxes.png"), dpi=120)
        fig, ax = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
        for d in days:
            h = eco_index(d, len(theta_e))
            l = ax[0].plot(prof["theta"][d], prof["depth"], label=f"day {d} ATS")
            ax[0].plot(theta_e[h, layers], mid_e, "--", color=l[0].get_color())
            ax[1].plot(prof["temperature"][d], prof["depth"], color=l[0].get_color())
            ax[1].plot(e["TEMP_vr"][h, layers], mid_e, "--", color=l[0].get_color())
        ax[0].invert_yaxis()
        ax[0].set_xlabel("theta [m3 m-3] (solid ATS, dashed EcoSIM)")
        ax[1].set_xlabel("T [C]")
        ax[0].set_ylabel("depth [m]")
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(eco_run, "comparison_profiles.png"), dpi=120)
        plot_saturation(plt, prof, e, layers, mid_e, days, os.path.join(eco_run, "comparison_saturation.png"))
        plot_temperature(plt, prof, e, layers, mid_e, days, os.path.join(eco_run, "comparison_temperature.png"))
        plot_snow(plt, ts, snow_ats, snow_e, days, os.path.join(eco_run, "comparison_snow.png"))
        lines += ["", "Plots: `comparison_fluxes.png` (daily fluxes, snow), `comparison_profiles.png` "
                  "(theta, T), `comparison_saturation.png` (saturation profiles), `comparison_temperature.png` "
                  "(temperature profiles), `comparison_snow.png` (snow depth)."]
    except ImportError:
        pass

    open(os.path.join(eco_run, "comparison.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
