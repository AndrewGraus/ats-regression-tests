#!/usr/bin/env python3
"""Generate a standalone EcoSIM column matched to an ATS-EcoSIM column.

From ecosim_prescribed_phenology.xml and the daymet forcing it uses, write:
  ecosim_prescribed_phenology_standalone_match.xml   ATS variant (13_ecosim/)
  data/standalone_match/column_graded19.exo          its column mesh
  data/standalone_match/grid_match.nc                EcoSIM grid, site and soil
  data/standalone_match/clim_match.nc                EcoSIM hourly climate
  data/standalone_match/pft_match.nc                 EcoSIM PFT management
  data/standalone_match/fatm_hist_GHGs_1750-2023.nc  EcoSIM GHG history (copy)
  data/standalone_match/standalone_match.namelist    EcoSIM namelist
  data/standalone_match/README.md                    what is and is not matched

Both models use prescribed phenology: the same EcoSIM plant code, plant
type (PFT table row 1, gr3s32), canopy (standalone's hard-coded LAI/SAI),
column, soil parameters (as far as EcoSIM's retention allows), forcing and
closed boundaries. See README.md for the differences that remain.

Usage (after `source set_dirs_debug.sh` from the workspace root):
  python make_standalone_match.py
The EcoSIM initial water content is the analytic ATS hydrostatic state;
compare_standalone_match.py checks it against the ATS initial checkpoint.
"""

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import netCDF4

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "standalone_match")
ATS_INPUT = os.path.join(HERE, "ecosim_prescribed_phenology.xml")
ATS_VARIANT = os.path.join(HERE, "ecosim_prescribed_phenology_standalone_match.xml")
# hourly forcing with site/calendar datasets, from tools/utils/daymet_to_ats_ecosim.py;
# the ATS input reads its forcing and site/calendar from the same file
FORCING = os.path.join(HERE, "data", "daymet_TaylorRiverCO_hourly.h5")
PFT_PARAMETERS = os.path.join(HERE, "data", "ecosim_pftpar_20260303.nc")

# ---------------------------------------------------------------------------
# column and run
# ---------------------------------------------------------------------------
NLAYERS = 19          # standalone EcoSIM: NL+1 must be <= JZ = 20 (Hour1Mod.F90:370)
DEPTH = 25.0          # m, as the ATS test
DZ_TOP = 0.05         # m, thickness of the top layer
with h5py.File(FORCING, "r") as _f:
    SITE = {k: float(_f[k][0]) for k in ("latitude [deg]", "solar noon [h]", "start year",
                                          "start day of year [0-364]", "start hour [h]")}
assert SITE["start day of year [0-364]"] == 0 and SITE["start hour [h]"] == 0, \
    "standalone runs start at 00:00 on 1 January"
YEAR = int(SITE["start year"])   # first model year in both runs (day 0 = 1 January)
SITE_LATITUDE = SITE["latitude [deg]"]
SITE_SOLAR_NOON = SITE["solar noon [h]"]
NDAYS = 365
PFT = "gr3s32"        # PFT table row 1 = ATS "vegetation type" 1
KOPPEN = 32           # selects gr3s32 in standalone (pft_type[0:4] + Koppen)
PARTICLE_DENSITY = 2.66  # Mg m-3, EcoSIM porosity = 1 - BKDSI/2.66 (no organic matter)

# values hard-coded in the coupled EcoSIM path (f90src/ATSUtils), matched here
COUPLED_SOIL_ALBEDO = 0.2    # ATSEcoSIMAdvanceMod.F90: SoilAlbedo_col = 0.2
# not set in the coupled path (allocation value 0, SetConstClimeForcing is
# standalone only); standalone gets the ATS surface-energy-balance height
WIND_HEIGHT = 3.0

# standalone PrescribePhenologyInterp (Prescribed_pheno/PrescribePhenolMod.F90
# lines 199-200 and 229-247): monthly tables, outside ATS coupling always
# evaluated at kmo=1, dofmon=1, ndaysmon=31, i.e. a fixed Dec/Jan mix
SP_LAI = [1.1852, 1.1821, 1.1554, 1.2433, 1.2922, 1.3341, 1.2296, 1.4118, 1.4343, 1.3941,
          1.2721, 1.2218]
SP_SAI = [0.3190, 0.3058, 0.3058, 0.3032, 0.3058, 0.3117, 0.3433, 0.3032, 0.3084, 0.3292,
          0.3656, 0.3249]


def standalone_canopy():
    kmo, dofmon, ndaysmon = 1, 1, 31
    t = (dofmon - 0.5) / ndaysmon
    it1 = int(t + 0.5)
    it2 = it1 + 1
    m1 = kmo + it1 - 1
    m2 = kmo + it2 - 1
    if m1 < 1: m1 = 12
    if m2 > 12: m2 = 1
    w1 = (it1 + 0.5) - t
    w2 = 1.0 - w1
    return (w1 * SP_LAI[m1 - 1] + w2 * SP_LAI[m2 - 1], w1 * SP_SAI[m1 - 1] + w2 * SP_SAI[m2 - 1])


# ---------------------------------------------------------------------------
# ATS input
# ---------------------------------------------------------------------------
def read_xml(path):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser)


def plist(node, *names):
    for name in names:
        found = [c for c in node if c.tag == "ParameterList" and c.get("name") == name]
        if len(found) != 1:
            raise KeyError(f"ParameterList {name!r} not found under {node.get('name')!r}")
        node = found[0]
    return node


def param(node, name):
    found = [c for c in node if c.tag == "Parameter" and c.get("name") == name]
    if len(found) != 1:
        raise KeyError(f"Parameter {name!r} not found under {node.get('name')!r}")
    return found[0]


def constant_value(evaluator):
    """value of a single function-constant in an independent-variable evaluator"""
    vals = [p for p in evaluator.iter("Parameter") if p.get("name") == "value"]
    if len(vals) != 1:
        raise ValueError(f"expected one constant in {evaluator.get('name')!r}")
    return vals[0]


def ats_parameters(root):
    pks = plist(root, "PKs")
    ev = plist(root, "state", "evaluators")
    wrm = plist(root, "state", "model parameters", "WRM parameters", "mineral")
    flow_ic = plist(pks, "subsurface flow", "initial conditions")
    eco = plist(pks, "EcoSIM for ATS")
    gravity = [float(x) for x in param(plist(root, "state", "initial conditions", "gravity"),
                                       "value").get("value").strip("{}").split(",")]
    p = dict(
        porosity=float(constant_value(plist(ev, "base_porosity")).get("value")),
        permeability=float(constant_value(plist(ev, "permeability")).get("value")),
        rock_density=float(constant_value(plist(ev, "density_rock")).get("value")),
        vg_alpha=float(param(wrm, "van Genuchten alpha [Pa^-1]").get("value")),
        vg_n=float(param(wrm, "van Genuchten n [-]").get("value")),
        sr=float(param(wrm, "residual saturation [-]").get("value")),
        head=float(param(flow_ic, "hydrostatic head [m]").get("value")),
        rho=float(param(flow_ic, "hydrostatic water density [kg m^-3]").get("value")),
        g=-gravity[2],
        T0=float(param(plist(pks, "subsurface energy", "initial conditions"), "value").get("value")),
        psi_fc=float(param(eco, "field capacity [Mpa]").get("value")),
        psi_wp=float(param(eco, "wilting point [Mpa]").get("value")),
    )
    return p


# ---------------------------------------------------------------------------
# soil and column
# ---------------------------------------------------------------------------
def layer_thicknesses():
    lo, hi = 1.0, 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        total = DZ_TOP * (r**NLAYERS - 1.0) / (r - 1.0)
        lo, hi = (r, hi) if total < DEPTH else (lo, r)
    dz = DZ_TOP * r**np.arange(NLAYERS)
    return dz * DEPTH / dz.sum()


def vg_saturation(pc, p):
    """ATS van Genuchten (Mualem m = 1 - 1/n), pc capillary pressure [Pa]"""
    m = 1.0 - 1.0 / p["vg_n"]
    se = (1.0 + (p["vg_alpha"] * np.maximum(pc, 0.0))**p["vg_n"])**(-m)
    return p["sr"] + (1.0 - p["sr"]) * se


def water_viscosity(T):
    """ATS viscosity_water.cc"""
    kav1, kbv1, kcv1, kbv2, kcv2, kT1 = 998.333, -8.1855, 0.00585, 1.3272, -0.001053, 293.15
    dT = kT1 - T
    if T < kT1:
        A = kav1 + (kbv1 + kcv1 * dT) * dT
        xi = 1301.0 * (1.0 / A - 1.0 / kav1)
    else:
        xi = (kbv2 + kcv2 * dT) * dT / (T - 168.15)
    return 0.001 * 10.0**xi


def soil(p, dz):
    depth_bottom = np.cumsum(dz)
    z_center = DEPTH - (depth_bottom - 0.5 * dz)          # ATS mesh: bottom z=0, top z=DEPTH
    s = dict(dz=dz, depth_bottom=depth_bottom, z_center=z_center)
    # EcoSIM derives porosity from bulk density with its own particle density, so
    # match porosity rather than the ATS rock density
    s["bulk_density"] = PARTICLE_DENSITY * (1.0 - p["porosity"])                # Mg m-3
    s["fc"] = p["porosity"] * vg_saturation(-p["psi_fc"] * 1e6, p)
    s["wp"] = p["porosity"] * vg_saturation(-p["psi_wp"] * 1e6, p)
    mu = water_viscosity(p["T0"])
    s["ksat_mm_h"] = p["permeability"] * p["rho"] * p["g"] / mu * 1000.0 * 3600.0
    # ATS "hydrostatic head" is relative to the surface: water table at DEPTH + head
    pc = p["rho"] * p["g"] * (z_center - (DEPTH + p["head"]))  # p = p_atm + rho g (z_wt - z)
    s["theta0"] = p["porosity"] * vg_saturation(pc, p)
    s["theta0_source"] = "analytic ATS hydrostatic state (head %g m)" % p["head"]
    return s


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def write_mesh(dz, path):
    sys.path.append(os.path.join(os.environ["ATS_SRC_DIR"], "tools", "meshing", "meshing_ats"))
    import meshing_ats
    m2 = meshing_ats.Mesh2D.from_Transect(np.array([0.0, 1.0]), np.array([DEPTH, DEPTH]))
    m3 = meshing_ats.Mesh3D.extruded_Mesh2D(m2, ["constant"] * NLAYERS, list(dz),
                                            [1] * NLAYERS, [1001] * NLAYERS)
    if os.path.exists(path):
        os.remove(path)
    m3.write_exodus(path)


def write_ats_variant(tree, lai, sai, dz, mesh_rel):
    root = tree.getroot()
    # mesh
    dom = plist(root, "mesh", "domain")
    for c in list(dom):
        if c.get("name") in ("mesh type", "generate mesh parameters"):
            dom.remove(c)
    mt = ET.Element("Parameter", name="mesh type", type="string", value="read mesh file")
    rm = ET.Element("ParameterList", name="read mesh file parameters", type="ParameterList")
    ET.SubElement(rm, "Parameter", name="file", type="string", value=mesh_rel)
    ET.SubElement(rm, "Parameter", name="format", type="string", value="Exodus II")
    dom.insert(0, rm)
    dom.insert(0, mt)
    # regions
    regions = plist(root, "regions")
    sub = plist(regions, "subsurface domain")
    for c in list(sub):
        sub.remove(c)
    ET.SubElement(sub, "ParameterList", name="region: all", type="ParameterList")
    top = plist(regions, "top_cell_obs", "region: box")
    param(top, "low coordinate").set("value", "{-10.0, -10.0, %.6f}" % (DEPTH - dz[0]))
    param(top, "high coordinate").set("value", "{10.0, 10.0, %.6f}" % DEPTH)
    # EcoSIM PK
    eco = plist(root, "PKs", "EcoSIM for ATS")
    # site and calendar: inherited from the input, read from the forcing file
    for name in ("starting year", "starting day of year [0-364]", "latitude [degrees]", "solar noon [h]"):
        assert plist(eco, name).find("Parameter[@name='file']") is not None, name
    param(eco, "EcoSIM precipitation").set("value", "true")
    # canopy and precipitation
    ev = plist(root, "state", "evaluators")
    constant_value(plist(ev, "surface-LAI")).set("value", "%.6f" % lai)
    constant_value(plist(ev, "surface-SAI")).set("value", "%.6f" % sai)
    # the input declares precipitation_total as a constant 0 (unused while EcoSIM
    # gets rain and snow separately); EcoSIM now gets rain + snow and partitions it
    tot = plist(ev, "surface-precipitation_total")
    for c in list(tot):
        tot.remove(c)
    ET.SubElement(tot, "Parameter", name="evaluator type", type="string", value="additive evaluator")
    ET.SubElement(tot, "Parameter", name="dependencies", type="Array(string)",
                  value="{surface-precipitation_rain, surface-precipitation_snow}")
    # daily visualization
    for name in ("domain", "surface domain", "surface"):
        param(plist(root, "visualization", name), "times start period stop").set(
            "value", "{0.0, 86400.0, -1.0}")
    header = (f"<!--\nATS-EcoSIM column matched to a standalone EcoSIM run. Generated by\n"
              f"make_standalone_match.py from ecosim_prescribed_phenology.xml; do not edit.\n"
              f"{NLAYERS} graded layers over {DEPTH:g} m, standalone canopy LAI {lai:.6f} and\n"
              f"SAI {sai:.6f}, start 1 January {YEAR}, EcoSIM partitions precipitation.\n"
              f"See data/standalone_match/README.md.\n-->\n\n")
    with open(ATS_VARIANT, "w") as f:
        ET.indent(root, space="  ")
        f.write(header + ET.tostring(root, encoding="unicode") + "\n")


def copy_structure(src, dst, nyear=None):
    """new NETCDF3_CLASSIC file with the dimensions, variables and attributes of src"""
    s = netCDF4.Dataset(src)
    d = netCDF4.Dataset(dst, "w", format=s.file_format)
    d.setncatts({k: s.getncattr(k) for k in s.ncattrs()})
    for name, dim in s.dimensions.items():
        size = None if dim.isunlimited() else len(dim)
        d.createDimension(name, size)
    for name, v in s.variables.items():
        fill = v.getncattr("_FillValue") if "_FillValue" in v.ncattrs() else None
        nv = d.createVariable(name, v.dtype, v.dimensions, fill_value=fill)
        nv.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})
    return s, d


def write_grid(path, s, p):
    src = os.path.join(os.environ["AMANZI_TPLS_BUILD_DIR"], "ecosim", "ecosim-0.1.0-source",
                       "examples", "inputs", "blodgett", "Blodget_grid_20251115.nc")
    t, d = copy_structure(src, path)
    for name, v in t.variables.items():                       # start from the template
        d[name][...] = v[...]
    t.close()
    n = NLAYERS
    def layers(values):
        a = np.zeros(d.dimensions["nlevs"].size)
        a[:n] = values
        return a[None, :]
    set_ = {
        "ALATG": SITE_LATITUDE, "ATCAG": p["T0"] - 273.15, "IETYPG": KOPPEN, "IDTBLG": 0,
        "RCHQNG": 0, "RCHQEG": 0, "RCHQSG": 0, "RCHQWG": 0,
        "RCHGNUG": 0, "RCHGEUG": 0, "RCHGSUG": 0, "RCHGWUG": 0, "RCHGDG": 0,
        "ASPX": 0, "SL0": 0, "DPTHSX": 0, "ALBS": COUPLED_SOIL_ALBEDO,
        "PSIFC": p["psi_fc"], "PSIWP": p["psi_wp"],
        "NUI": 1, "NJ": n, "NL1": 0, "NL2": 0,
    }
    for k, v in set_.items():
        d[k][...] = v
    d["DHI"][...] = 1.0
    d["DVI"][...] = 1.0
    d["CDPTH"][...] = layers(s["depth_bottom"])
    d["BKDSI"][...] = layers(s["bulk_density"])
    d["FC"][...] = layers(s["fc"])
    d["WP"][...] = layers(s["wp"])
    d["SCNV"][...] = layers(s["ksat_mm_h"])
    d["SCNH"][...] = layers(s["ksat_mm_h"])
    d["FHOL"][...] = layers(0.0)
    d["ROCK"][...] = layers(0.0)
    d["THW"][...] = layers(s["theta0"])
    d["THI"][...] = layers(-1.0)     # no ice; THI = 0 means ice at WP (StartsMod.F90:489)
    # texture: a fixed loam (not matched: ATS thermal properties are separate)
    d["CSAND"][...] = layers(400.0)
    d["CSILT"][...] = layers(400.0)
    # mineral soil: no organic matter or litter (as the coupled path, CSoilOrgM = 0)
    for k in ["CORGC", "CORGR", "CORGN", "CORGP", "RSCfL", "RSNfL", "RSPfL", "RSCwL",
              "RSNwL", "RSPwL", "RSCmL", "RSNmL", "RSPmL"]:
        d[k][...] = layers(0.0)
    for k in ["RSCf", "RSNf", "RSPf", "RSCw", "RSNw", "RSPw", "RSCm", "RSNm", "RSPm"]:
        d[k][...] = 0.0
    d.close()


def forcing_hourly(p):
    # The coupled EcoSIM call at the start of an hour gets the ATS forcing there
    # (the value stamped at the start of the hour, the mean of that hour), and
    # computes that hour. Standalone hour J of day d covers the same hour, so it
    # gets record 24 (d - 1) + J - 1.
    with h5py.File(FORCING, "r") as f:
        n = 366 * 24
        interp = {k: f[k][:n] for k in f if f[k].shape[0] > 1 and k != "time [s]"}
        assert np.allclose(f["time [s]"][:n], 3600.0 * np.arange(n)), "hourly records from time 0"
    shape = (1, 366, 24, 1)
    out = {
        "TMPH": (interp["air temperature [K]"] - 273.15).reshape(shape),
        "WINDH": interp["wind speed [m s^-1]"].reshape(shape),
        "RAINH": ((interp["precipitation rain [m s^-1]"] + interp["precipitation snow [m SWE s^-1]"])
                  * 1000.0 * 3600.0).reshape(shape),
        "DWPTH": (interp["vapor pressure air [Pa]"] / 1000.0).reshape(shape),
        "SRADH": interp["incoming shortwave radiation [W m^-2]"].reshape(shape),
    }
    return out


def write_climate(path, p):
    src = os.path.join(os.environ["AMANZI_TPLS_BUILD_DIR"], "ecosim", "ecosim-0.1.0-source",
                       "examples", "inputs", "blodgett", "Blodget.clim.2012-2022.nc")
    t, d = copy_structure(src, path)
    t.close()
    for k, v in forcing_hourly(p).items():
        d[k][0:1, ...] = v
    d["year"][0:1] = [YEAR]
    d["Z0G"][0:1, :] = WIND_HEIGHT
    d["IFLGW"][0:1, :] = 0
    d["ZNOONG"][0:1, :] = SITE_SOLAR_NOON
    d["PHRG"][0:1, :] = 7.0
    for k in ["CN4RIG", "CNORIG", "CPORG", "CALRG", "CFERG", "CCARG", "CMGRG", "CNARG", "CKARG",
              "CSORG", "CCLRG"]:
        d[k][0:1, :] = 0.0
    d.close()


def chars(text, n):
    a = np.zeros(n, "S1")
    a[:] = b" "
    a[:len(text)] = np.array(list(text), "S1")
    return a


def write_pft(path):
    src = os.path.join(os.environ["AMANZI_TPLS_BUILD_DIR"], "ecosim", "ecosim-0.1.0-source",
                       "examples", "inputs", "blodgett", "Blodget_pft_20251112.ndlf.nc")
    t, d = copy_structure(src, path)
    for k in ["NH1", "NV1", "NH2", "NV2", "NZ"]:
        d[k][...] = t[k][...]
    t.close()
    d["pft_dflag"].assignValue(0)
    nm = np.ma.masked_all((1, 1, 5), "i2")
    nm[0, 0, 0] = 0
    d["nmgnts"][0:1, ...] = nm
    ptype = np.zeros((1, 1, 5, 10), "S1")
    ptype[0, 0, 0, :] = chars(PFT, 10)
    d["pft_type"][0:1, ...] = ptype
    plt = np.zeros((1, 1, 5, 128), "S1")
    plt[0, 0, 0, :] = chars("01019999 0.6 0.004", 128)        # population ignored in SP mode
    d["pft_pltinfo"][0:1, ...] = plt
    d["pft_mgmt"][0:1, ...] = np.zeros((1, 1, 5, 24, 128), "S1")
    d.close()


NAMELIST = """&regression_test
 cells = 1
/

&ecosim
case_name          = 'standalone_match'
do_regression_test = .false.
pft_file_in        = '{rel}/../ecosim_pftpar_20260303.nc'
grid_file_in       = '{rel}/grid_match.nc'
pft_mgmt_in        = '{rel}/pft_match.nc'
clm_hour_file_in   = '{rel}/clim_match.nc'
clm_factor_in      = 'NO'
soil_mgmt_in       = 'NO'
atm_ghg_in         = '{rel}/fatm_hist_GHGs_1750-2023.nc'

lverbose        = .false.
ldo_sp_mode     = .true.
disp_planttrait = .true.
plant_model     = .false.
microbial_model = .false.
soichem_model   = .false.

start_date   = '{year}0101000000'
grid_mode    = 3
continue_run = .false.
forc_periods = {year},{year},1

hist_fincl1 = 'PSI_vr','VSPore_vr','rIceFLP_vr','RootH2OUptake_vr'

NPXS=30,30,30
NPYS=10,10,10
NCYC_LITR=30
NCYC_SNOW=20
hist_mfilt  = 8784
hist_nhtfrq = 1
/

&bbgcforc
/

&Microbes
/

&ecosim_time
rest_opt    = 'never'
rest_frq    = 1
delta_time  = 3600.
stop_n      = {ndays}
stop_option = 'ndays'
diag_frq    = 1
diag_opt    = 'nsteps'
/
"""


def write_readme(path, p, s, lai, sai):
    rows = "\n".join(f"| {i+1} | {s['dz'][i]:.4f} | {s['depth_bottom'][i]:.4f} | {s['theta0'][i]:.5f} |"
                     for i in range(NLAYERS))
    text = f"""# Standalone EcoSIM column matched to the ATS-EcoSIM column

Generated by `13_ecosim/make_standalone_match.py`; do not edit by hand.

## Running
From the workspace root, after `source set_dirs_debug.sh`:

    cd $ATS_SRC_DIR/testing/ats-regression-tests/13_ecosim
    python make_standalone_match.py                      # these files and the ATS variant
    mkdir -p ats_match && cd ats_match
    $ATS_DIR/bin/ats --xml_file=../ecosim_prescribed_phenology_standalone_match.xml > log 2>&1
    cd .. && mkdir -p ecosim_match && cd ecosim_match
    $AMANZI_TPLS_DIR/ecosim/bin/ecosim.f90.x ../data/standalone_match/standalone_match.namelist > log 2>&1
    cd .. && python compare_standalone_match.py ats_match ecosim_match

The namelist uses relative paths: EcoSIM truncates file names to 128
characters (`f90src/Utils/ncdio_pio.F90`).

## Matched
| item | value |
|---|---|
| plant code | prescribed phenology (`ldo_sp_mode`), same EcoSIM path as the coupling |
| plant type | `{PFT}` (PFT table row 1, ATS vegetation type 1; standalone `pft_type` + Koppen {KOPPEN}) |
| canopy | LAI {lai:.6f}, SAI {sai:.6f} (standalone hard-coded values), height 17 m, root type 1, population 0.6 |
| column | {NLAYERS} layers over {DEPTH:g} m, 1 m2, top layer {s['dz'][0]:.3f} m |
| porosity | {p['porosity']:g}: EcoSIM bulk density {s['bulk_density']:.4f} Mg m-3 with its particle density {PARTICLE_DENSITY:g} (ATS rock density {p['rock_density']:g} kg m-3 is not matched) |
| retention | FC = {s['fc']:.5f} at {p['psi_fc']:g} MPa, WP = {s['wp']:.5f} at {p['psi_wp']:g} MPa, from the ATS van Genuchten curve (alpha {p['vg_alpha']:g} Pa-1, n {p['vg_n']:g}, sr {p['sr']:g}) |
| saturated conductivity | {s['ksat_mm_h']:.4f} mm h-1 = k rho g / mu, k = {p['permeability']:g} m2, mu at {p['T0']:g} K (ATS viscosity_water) |
| boundaries | closed: standalone `grid_mode=3` (no lateral flow or runoff), `RCHGDG=0` (no-flow bottom), no water table; ATS has no flow BCs |
| initial water | {s['theta0_source']} |
| forcing | hourly forcing with a daily cycle from `daymet_to_ats_ecosim.py` (`data/daymet_TaylorRiverCO_hourly.h5`); standalone hour J of day d = the record at the start of that hour, which the coupled call at that time receives; precipitation = rain + snow, partitioned by EcoSIM in both runs (ATS variant: "EcoSIM precipitation" = true) |
| dates | both start on 1 January {YEAR}, {NDAYS} days, hourly steps |
| site | latitude {SITE_LATITUDE:g} and solar noon {SITE_SOLAR_NOON:g} h from the forcing file (ATS PK file parameters, standalone ALATG/ZNOONG); soil albedo {COUPLED_SOIL_ALBEDO:g} as hard-coded in the coupled path; no organic matter or litter |

## Not matched
- **Soil water physics:** EcoSIM's own soil water solver and its retention curve, which agrees with van Genuchten only at FC and WP, against ATS Richards with van Genuchten. In the coupled run, ATS's soil state replaces EcoSIM's every hour.
- **Micropore conductivity:** EcoSIM has no temperature correction; ATS uses its temperature-dependent viscosity. Ksat is matched at {p['T0']:g} K.
- **Bottom heat boundary:** ATS fixes {p['T0']:g} K; EcoSIM applies a geothermal heat flux. Both start at {p['T0'] - 273.15:.2f} C (EcoSIM: mean annual temperature `ATCAG`, no initial ice).
- **Thermal properties:** EcoSIM derives them from texture (a fixed loam here); ATS uses the three-phase wet/dry model.
- **Hard-coded in the coupled path:** FC/WP water contents 0.5/0.25, root-uptake conductivity 0.000571, canopy water potential -2 MPa and canopy temperature reset each hour.
- **Wind measurement height:** never set in the coupled path (0); standalone gets {WIND_HEIGHT:g} m.

## Layers
| layer | dz [m] | depth to bottom [m] | initial theta [m3 m-3] |
|---|---|---|---|
{rows}
"""
    open(path, "w").write(text)


def check_file(path, expect):
    d = netCDF4.Dataset(path)
    for name, (lo, hi) in expect.items():
        v = np.ma.filled(d[name][...].astype(float), np.nan)
        v = v[np.isfinite(v)]
        if v.size and (v.min() < lo or v.max() > hi):
            raise ValueError(f"{path}:{name} range {v.min():g}..{v.max():g} outside {lo:g}..{hi:g}")
    d.close()


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()

    os.makedirs(OUT, exist_ok=True)
    tree = read_xml(ATS_INPUT)
    p = ats_parameters(tree.getroot())
    lai, sai = standalone_canopy()
    dz = layer_thicknesses()
    s = soil(p, dz)

    assert abs(dz.sum() - DEPTH) < 1e-9 and len(dz) == NLAYERS
    assert p["porosity"] > s["fc"] > s["wp"] > p["porosity"] * p["sr"], (s["fc"], s["wp"])
    assert np.all((s["theta0"] > 0.0) & (s["theta0"] < 1.0)), "THW must be strictly in (0,1)"

    print(f"layers: {NLAYERS}, dz {dz[0]:.4f} .. {dz[-1]:.4f} m")
    print(f"FC {s['fc']:.5f}  WP {s['wp']:.5f}  Ksat {s['ksat_mm_h']:.4f} mm/h  "
          f"bulk density {s['bulk_density']:.4f}")
    print(f"canopy LAI {lai:.6f} SAI {sai:.6f}")
    print(f"initial theta {s['theta0'].min():.4f} .. {s['theta0'].max():.4f} ({s['theta0_source']})")

    mesh = os.path.join(OUT, f"column_graded{NLAYERS}.exo")
    write_mesh(dz, mesh)
    write_ats_variant(read_xml(ATS_INPUT), lai, sai, dz,
                      os.path.join("..", "data", "standalone_match", os.path.basename(mesh)))
    write_grid(os.path.join(OUT, "grid_match.nc"), s, p)
    write_climate(os.path.join(OUT, "clim_match.nc"), p)
    write_pft(os.path.join(OUT, "pft_match.nc"))
    shutil.copy(os.path.join(os.environ["AMANZI_TPLS_BUILD_DIR"], "ecosim", "ecosim-0.1.0-source",
                             "input_data", "fatm_hist_GHGs_1750-2023.nc"), OUT)
    rel = os.path.join("..", "data", "standalone_match")
    open(os.path.join(OUT, "standalone_match.namelist"), "w").write(
        NAMELIST.format(rel=rel, year=YEAR, ndays=NDAYS))
    write_readme(os.path.join(OUT, "README.md"), p, s, lai, sai)

    check_file(os.path.join(OUT, "grid_match.nc"),
               {"CDPTH": (0, DEPTH + 1e-6), "FC": (0, 1), "WP": (0, 1), "THW": (0, 1)})
    check_file(os.path.join(OUT, "clim_match.nc"),
               {"TMPH": (-60, 50), "SRADH": (0, 1400), "DWPTH": (0, 7), "RAINH": (0, 100),
                "WINDH": (0, 50)})
    print("wrote", ATS_VARIANT)
    for f in sorted(os.listdir(OUT)):
        print("wrote", os.path.join(OUT, f))


if __name__ == "__main__":
    main()
