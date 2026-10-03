# Water-balance reference: old source evaluators

This directory holds the output of a 200-day `ecosim_freeze_thaw` run made
with ATS **before** the EcoSIM source fields became PK-owned primary
evaluators. Before that fix, the input declared `surface-ecosim_water_source` and
`surface-ecosim_source` as `"independent variable constant"` (value 0). The
evaluator reported "changed" only once, so overland flow kept using the cached
zero for `surface-ecosim_water_source_mols`, and the water EcoSIM computed
never reached ATS.

Check it without rebuilding anything:

    python ../../../check_water_balance.py .

(run from this directory, or pass the directory path from `13_ecosim`).

Expected result (FAIL):

    .: 4800 steps, 4800 EcoSIM calls, t = 0 .. 1.728e+07 s, area 1 m2
       storage change         -4.31000080e-05 mol   -7.7646e-07 mm
       applied by ATS          0.00000000e+00 mol             0 mm
       reported by EcoSIM     -1.58968838e+04 mol       -286.39 mm   (pending at end: -0.21083 mm)
       R1 ATS conservation     -2.711e-09   (residual -4.310e-05 mol, without lag: -2.711e-09)
       R2 coupling              1.000e+00
       FAIL

- **R1 ≈ 0:** ATS conserved the water it was given.
- **R2 = 1 (100 % of the reported water missing):** it was given none of the water EcoSIM reported.

Compare it with the current code run on the current `13_ecosim/ecosim_freeze_thaw.xml`
(the old input is rejected by the current PK), where R1 and R2 are both
below 1e-6.

## Contents

| file | |
|---|---|
| `ecosim_freeze_thaw_old_input.xml` | The exact input used. It is `git show 8b9bcea:13_ecosim/ecosim_freeze_thaw.xml` (with the two `"independent variable constant"` source declarations) plus the same `observations` block as the current `13_ecosim` inputs. |
| `water_balance_surface.dat`, `water_balance_subsurface.dat` | The observations read by `check_water_balance.py`. |
| `ats_version.txt` | The repository commits used. |

## Re-running the old code

The relevant change touches only `src/pks/ecosim/EcoSIM_ATS_interface.{cc,hh}`
in ATS. The last commit before it is `c80871eb` on `smolins/ecosim_state_fix`.
EcoSIM and Amanzi can stay at the versions in `ats_version.txt`. Everything is
run from the workspace root after `source set_dirs_debug.sh`.

### Option A: temporarily check out the two old files (one-off)

```sh
source set_dirs_debug.sh
cd $ATS_SRC_DIR
git status --short src/pks/ecosim          # must be clean, or stash first
git checkout c80871eb -- src/pks/ecosim/EcoSIM_ATS_interface.cc \
                         src/pks/ecosim/EcoSIM_ATS_interface.hh
(cd $AMANZI_BUILD_DIR && make -j12 install)

cd $ATS_SRC_DIR/testing/ats-regression-tests/13_ecosim
mkdir wb_old_source_evaluator && cd wb_old_source_evaluator   # inputs use ../data
cp ../data/water_balance_reference/old_source_evaluator/ecosim_freeze_thaw_old_input.xml .
$ATS_DIR/bin/ats --xml_file=ecosim_freeze_thaw_old_input.xml > log 2>&1
python ../check_water_balance.py .

# restore the current code and rebuild
cd $ATS_SRC_DIR
git checkout HEAD -- src/pks/ecosim/EcoSIM_ATS_interface.cc \
                     src/pks/ecosim/EcoSIM_ATS_interface.hh
git status --short src/pks/ecosim          # must be clean again
(cd $AMANZI_BUILD_DIR && make -j12 install)
```

The `git checkout <commit> -- <files>` form stages the old files. The
`git checkout HEAD -- <files>` at the end undoes both the staging and the
working-tree change.

### Option B: a separate source tree and build (when the old code is needed repeatedly)

This keeps a permanent old build next to the current one, so neither has to be
rebuilt when switching. ATS is the `src/physics/ats` submodule of Amanzi, so
the old tree is an Amanzi worktree with an ATS worktree inside it. It reuses
the existing TPL install, which contains EcoSIM. This recipe has not been run
yet; adjust it if `bootstrap.sh` options change.

```sh
source set_dirs_debug.sh
OLD=$ATS_BASE/repos/amanzi-old-source-evaluator
git -C $AMANZI_SRC_DIR worktree add $OLD 42cadd935
git -C $ATS_SRC_DIR worktree add $OLD/src/physics/ats c80871eb   # fills the empty submodule dir

# --enable-ats_dev keeps bootstrap from re-cloning or switching the ATS branch;
# --tpl-config-file reuses the installed TPLs instead of rebuilding them
$OLD/bootstrap.sh --debug --with-mpi=$OPENMPI_DIR --enable-shared \
   --enable-clm --enable-ecosim --disable-structured --enable-unstructured \
   --enable-mesh_mstk --disable-mesh_moab --enable-hypre --disable-silo \
   --disable-petsc --disable-amanzi_physics --enable-ats_physics \
   --enable-ats_dev --disable-geochemistry --disable-superlu \
   --tpl-config-file=$AMANZI_TPLS_DIR/share/cmake/amanzi-tpl-config.cmake \
   --amanzi-build-dir=$ATS_BASE/amanzi-build-old-source-evaluator \
   --amanzi-install-prefix=$ATS_BASE/amanzi-install-old-source-evaluator \
   --with-cmake=`which cmake` --with-ctest=`which ctest` --parallel=12
```

The options match `build_ATS_generic.sh`. Then run the input as in Option A,
but with `$ATS_BASE/amanzi-install-old-source-evaluator/bin/ats`. To remove the
old tree, run `git -C $ATS_SRC_DIR worktree remove $OLD/src/physics/ats`, then
`git -C $AMANZI_SRC_DIR worktree remove $OLD`, then delete the two old build
and install directories.

## Why a stored reference

The gold comparison in `ecosim.cfg` checks state fields only, against gold that
was generated with the bug, so it could not detect the missing water. This
reference records what the failure looks like in the water budget, so that
`check_water_balance.py` can be shown to detect it without rebuilding the old
code.
