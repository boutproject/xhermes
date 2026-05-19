#!/usr/bin/env python3

import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy.testing as npt
import pytest
from matplotlib import pyplot as plt

import xhermes

###
# This test checks all the poloidal selections from slice_poloidal against 10 grids of
# different topologies and guard cell configurations. The expected R coordinates of the selected
# cells are stored in a JSON file and compared to the test output. The test can also be used to
# generate the expected data by setting generate_data = True, which will save the R coordinates
# of the selected cells for all grids to the JSON file. The test also generates plots of
# the logical and poloidal grid with the selected region highlighted for visual inspection.
# Optional developer workflows are controlled with local booleans below.
###

plot = False
generate_data = False
selection_check_enabled = True
remove_old_images = True

##################################################################################

POLOIDAL_RCOORDS_FILE = Path(__file__).parent / "test_slice_poloidal_reference.json"
RADIAL_RCOORDS_FILE = Path(__file__).parent / "test_slice_radial_reference.json"

if POLOIDAL_RCOORDS_FILE.exists():
    with open(POLOIDAL_RCOORDS_FILE) as f:
        poloidal_Rcoords = json.load(f)
else:
    poloidal_Rcoords = {}

if RADIAL_RCOORDS_FILE.exists():
    with open(RADIAL_RCOORDS_FILE) as f:
        radial_Rcoords = json.load(f)
else:
    radial_Rcoords = {}


@pytest.fixture(scope="session")
def all_grids(tmp_path_factory):
    # Pytest fixture to make temporary directory
    # the decorator makes sure the files are only used for this session
    # and don't break stuff
    tmp = tmp_path_factory.mktemp("hypnotoad")
    zip_path = tmp / "Hypnotoad_examples.zip"
    url = "https://zenodo.org/records/17966926/files/Hypnotoad_examples.zip"
    urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp)

    all_grids = dict(
        sn_grids=dict(
            lsn=xhermes.HypnotoadGrid(tmp / "example_lsn.grd.nc"),
            lsn_noguards=xhermes.HypnotoadGrid(tmp / "example_lsn_noguards.grd.nc"),
            usn=xhermes.HypnotoadGrid(tmp / "example_usn.grd.nc"),
            usn_noguards=xhermes.HypnotoadGrid(tmp / "example_usn_noguards.grd.nc"),
        ),
        dn_grids=dict(
            cdn=xhermes.HypnotoadGrid(tmp / "example_cdn.grd.nc"),
            cdn_noguards=xhermes.HypnotoadGrid(tmp / "example_cdn_noguards.grd.nc"),
            ldn=xhermes.HypnotoadGrid(tmp / "example_ldn.grd.nc"),
            ldn_noguards=xhermes.HypnotoadGrid(tmp / "example_ldn_noguards.grd.nc"),
            udn2=xhermes.HypnotoadGrid(tmp / "example_udn2.grd.nc"),
            udn2_noguards=xhermes.HypnotoadGrid(tmp / "example_udn2_noguards.grd.nc"),
        ),
    )

    all_grids["sn_grids"]["lsn_removeguards"] = all_grids["sn_grids"][
        "lsn"
    ].remove_guards()
    all_grids["dn_grids"]["cdn_removeguards"] = all_grids["dn_grids"][
        "cdn"
    ].remove_guards()

    return all_grids


def test_radial_slices(all_grids):
    radial_selections = [
        "domain",
        "domain_guards",
        "xguards",
        "sol",
        "sol_guards",
        "core",
        "core_guards",
        "pfr",
        "pfr_guards",
        "boundary_xlow",
        "boundary_guard_xlow",
        "boundary_xup",
        "boundary_guard_xup",
    ]

    grids = dict(
        lsn=all_grids["sn_grids"]["lsn"],
        lsn_removeguards=all_grids["sn_grids"]["lsn_removeguards"],
        cdn=all_grids["dn_grids"]["cdn"],
        cdn_removeguards=all_grids["dn_grids"]["cdn_removeguards"],
    )

    if selection_check_enabled:
        # All grids have the same radial selectors, so we can just check one of them
        ds_test = all_grids["sn_grids"]["lsn"]
        available_selections = xhermes.selector_radial(ds_test, return_available=True)
        missing = [
            selection
            for selection in available_selections
            if selection not in radial_selections
        ]

        if missing:
            msg = "The following radial selectors are missing in the the test definition:\n"
            for selection in missing:
                msg += f'  "{selection}"\n'
            raise ValueError(msg)

    if plot:
        output_dir = Path(__file__).parent / "radial_selection_images"
        output_dir.mkdir(exist_ok=True)

        if remove_old_images:
            # Remove all old images if present
            for output_file in output_dir.glob("*.png"):
                output_file.unlink()

    nrows = len(grids)

    for selection in radial_selections:
        print(f"\n{selection}")
        print("--------------------")

        if plot:
            fig = plt.figure(figsize=(6, nrows * 4))
            gs = fig.add_gridspec(nrows, 2, width_ratios=[1, 1], hspace=0.5, wspace=0.1)

        for row, (name, grid) in enumerate(grids.items()):
            if "guard" in selection and "removeguards" in name:
                continue

            print(f"{name}, ", end="")
            radial_selector = xhermes.selector_radial(grid, selection)

            if plot:
                ax0 = fig.add_subplot(gs[row, 0])
                ax1 = fig.add_subplot(gs[row, 1])
                xhermes.plot_selection(
                    grid,
                    custom_selection=(radial_selector, slice(None)),
                    axes=[ax0, ax1],
                )
                ax0.set_title(f"{name}\nLogical")
                ax1.set_title(f"{name}\nPoloidal")

            R_test = grid["Rxy"][radial_selector, slice(None)]

            if generate_data:
                if selection not in radial_Rcoords:
                    radial_Rcoords[selection] = {}
                radial_Rcoords[selection][name] = R_test.tolist()
            else:
                if selection not in radial_Rcoords:
                    raise ValueError(
                        f"Selection {selection} not found in radial reference data, cannot test!"
                    )
                if name not in radial_Rcoords[selection]:
                    raise ValueError(
                        f"Grid {name} not found in radial reference data for selection {selection}, cannot test!"
                    )
                R_expected = radial_Rcoords[selection][name]
                npt.assert_array_equal(
                    R_test,
                    R_expected,
                    err_msg=f"Radial selector mismatch for {selection} in {name}!",
                )

        print()

        if plot:
            fig.savefig(
                output_dir / f"{selection}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    if generate_data:
        with open(RADIAL_RCOORDS_FILE, "w") as f:
            json.dump(radial_Rcoords, f, indent=2)
        print(f"Saved expected data to {RADIAL_RCOORDS_FILE}")

    if generate_data or plot:
        raise Exception(
            "Test is set to fail if generate_data = True or plot = True, to prevent accidentally leaving these on. Set both to False to run the test normally."
        )


def test_poloidal_slices(all_grids):
    poloidal_selections = {}

    poloidal_selections["sn"] = [
        "core",
        "inner_target",
        "outer_target",
        "inner_xpoint",
        "outer_xpoint",
        "targets",
        "inner_lower_midplane",
        "inner_upper_midplane",
        "outer_lower_midplane",
        "outer_upper_midplane",
        "inner_sol_extra",
        "outer_sol_extra",
        "inner_sol_extra_guards",
        "outer_sol_extra_guards",
        "inner_sol",
        "outer_sol",
        "inner_sol_guards",
        "outer_sol_guards",
        "sol",
        "sol_guards",
        "inner_divertor",
        "outer_divertor",
        "inner_divertor_guards",
        "outer_divertor_guards",
        "pfr",
        "pfr_guards",
        "yguards",
    ]

    poloidal_selections["dn"] = [
        "inner_lower_target",
        "outer_lower_target",
        "inner_upper_target",
        "outer_upper_target",
        "targets",
        "inner_lower_xpoint",
        "inner_upper_xpoint",
        "outer_upper_xpoint",
        "outer_lower_xpoint",
        "inner_lower_midplane",
        "inner_upper_midplane",
        "outer_upper_midplane",
        "outer_lower_midplane",
        "inner_lower_sol_extra",
        "inner_upper_sol_extra",
        "outer_upper_sol_extra",
        "outer_lower_sol_extra",
        "inner_lower_sol_extra_guards",
        "inner_upper_sol_extra_guards",
        "outer_upper_sol_extra_guards",
        "outer_lower_sol_extra_guards",
        "inner_lower_sol",
        "inner_upper_sol",
        "outer_upper_sol",
        "outer_lower_sol",
        "inner_lower_sol_guards",
        "inner_upper_sol_guards",
        "outer_upper_sol_guards",
        "outer_lower_sol_guards",
        "inner_lower_upstream",
        "inner_upper_upstream",
        "outer_upper_upstream",
        "outer_lower_upstream",
        "inner_lower_upstream_extra",
        "inner_upper_upstream_extra",
        "outer_upper_upstream_extra",
        "outer_lower_upstream_extra",
        "inner_lower_divertor",
        "inner_upper_divertor",
        "outer_upper_divertor",
        "outer_lower_divertor",
        "inner_lower_divertor_guards",
        "inner_upper_divertor_guards",
        "outer_upper_divertor_guards",
        "outer_lower_divertor_guards",
        "inner_sol",
        "outer_sol",
        "sol",
        "inner_sol_guards",
        "outer_sol_guards",
        "sol_guards",
        "inner_core",
        "outer_core",
        "core",
        "inner_lower_pfr",
        "inner_upper_pfr",
        "outer_lower_pfr",
        "outer_upper_pfr",
        "lower_pfr",
        "upper_pfr",
        "pfr",
        "inner_lower_pfr_guards",
        "inner_upper_pfr_guards",
        "outer_lower_pfr_guards",
        "outer_upper_pfr_guards",
        "lower_pfr_guards",
        "upper_pfr_guards",
        "pfr_guards",
        "yguards",
    ]

    if selection_check_enabled:
        # Test if all selectors are present in test set
        ds_test_sn = all_grids["sn_grids"]["lsn"]
        ds_test_dn = all_grids["dn_grids"]["cdn"]

        missing = dict(sn=[], dn=[])

        for topology_type, ds in zip(["sn", "dn"], [ds_test_sn, ds_test_dn]):
            available_selections = xhermes.selector_poloidal(ds, return_available=True)
            for selection in available_selections:
                if selection not in poloidal_selections[topology_type]:
                    missing[topology_type].append(selection)

        if len(missing["sn"]) > 0 or len(missing["dn"]) > 0:
            msg = "The following selectors are missing from the test grids:\n"
            for topology_type in ["sn", "dn"]:
                if len(missing[topology_type]) > 0:
                    msg += f"\n{topology_type}:\n"
                    for sel in missing[topology_type]:
                        msg += f'  "{sel}"\n'
            raise ValueError(msg)

    if plot:
        output_dir = Path(__file__).parent / "poloidal_selection_images"
        output_dir.mkdir(exist_ok=True)

        if remove_old_images:
            # Remove all old images if present
            for output_file in output_dir.glob("*.png"):
                output_file.unlink()

    for topology_type in ["sn", "dn"]:
        print(f"\n-> Testing {topology_type} selectors...")
        grids = all_grids[f"{topology_type}_grids"]
        nrows = len(grids)

        for i, selection in enumerate(poloidal_selections[topology_type]):
            print(f"\n{selection}")
            print("--------------------")

            if plot:
                fig = plt.figure(figsize=(6, nrows * 4))
                gs = fig.add_gridspec(
                    nrows, 2, width_ratios=[1, 1], hspace=0.5, wspace=0.1
                )

            for row, (grid_name, grid) in enumerate(grids.items()):
                if "guards" in selection and (
                    "noguards" in grid_name or "removeguards" in grid_name
                ):
                    continue
                print(f"{grid_name}, ", end="")
                poloidal_selector = xhermes.selector_poloidal(grid, selection)

                # Generate plots for visual check
                if plot:
                    ax0 = fig.add_subplot(gs[row, 0])
                    ax1 = fig.add_subplot(gs[row, 1])
                    xhermes.plot_selection(
                        grid,
                        custom_selection=(slice(None), poloidal_selector),
                        axes=[ax0, ax1],
                    )
                    ax0.set_title(f"{grid_name}\nLogical")
                    ax1.set_title(f"{grid_name}\nPoloidal")

                # Generate test data and compare to expected
                R_test = grid["Rxy"][slice(None), poloidal_selector]

                if generate_data:
                    if selection not in poloidal_Rcoords:
                        poloidal_Rcoords[selection] = {}
                    poloidal_Rcoords[selection][grid_name] = R_test.tolist()
                else:
                    if selection not in poloidal_Rcoords:
                        raise ValueError(
                            f"Selection {selection} not found in reference data, cannot test!"
                        )
                    if grid_name not in poloidal_Rcoords[selection]:
                        raise ValueError(
                            f"Grid {grid_name} not found in reference data for selection {selection}, cannot test!"
                        )
                    R_expected = poloidal_Rcoords[selection][grid_name]
                    npt.assert_array_equal(
                        R_test,
                        R_expected,
                        err_msg=f"Selector mismatch for {selection} in {grid_name}!",
                    )
            print()

            if plot:
                fig.savefig(
                    output_dir / f"{selection}-{topology_type}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

    if generate_data:
        with open(POLOIDAL_RCOORDS_FILE, "w") as f:
            json.dump(poloidal_Rcoords, f, indent=2)
        print(f"Saved expected data to {POLOIDAL_RCOORDS_FILE}")

    if generate_data or plot:
        raise Exception(
            "Test is set to fail if generate_data = True or plot = True, to prevent accidentally leaving these on. Set both to False to run the test normally."
        )
