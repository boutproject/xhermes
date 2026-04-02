#!/usr/bin/env python3

import json
import xhermes
import pytest
import numpy as np
import numpy.testing as npt
from pathlib import Path
from matplotlib import pyplot as plt
import textwrap
from urllib.request import urlretrieve
import zipfile
from time import time


###
# This test checks all the poloidal selections from slice_poloidal against 10 grids of 
# different topologies and guard cell configurations. The expected R coordinates of the selected
# cells are stored in a JSON file and compared to the test output. The test can also be used to
# generate the expected data by setting generate_data = True, which will save the R coordinates
# of the selected cells for all grids to the JSON file. The test also generates plots of
# the logical and poloidal grid with the selected region highlighted for visual inspection, which
# can be saved by setting plot = True. 
###

RCOORDS_FILE = Path(__file__).parent / "test_slice_poloidal_reference.json"

if RCOORDS_FILE.exists():
    with open(RCOORDS_FILE) as f:
        Rcoords = json.load(f)
else:
    Rcoords = {}


@pytest.fixture(scope="session")
def example_grids(tmp_path_factory):
    
    # Pytest fixture to make temporary directory
    # the decorator makes sure the files are only used for this session
    # and don't break stuff
    tmp = tmp_path_factory.mktemp("hypnotoad")
    zip_path = tmp / "Hypnotoad_examples.zip"
    url = "https://zenodo.org/records/17966926/files/Hypnotoad_examples.zip"
    urlretrieve(url, "Hypnotoad_examples.zip")
    with zipfile.ZipFile("Hypnotoad_examples.zip", 'r') as z:
        z.extractall(tmp)
        
    return tmp

poloidal_selections = {}

poloidal_selections["sn"] = [
        "inner_target",
        "outer_target",
        "inner_xpoint",
        "outer_xpoint",
        "inner_sol",
        "outer_sol",
        "inner_sol_extra",
        "outer_sol_extra",
        "sol",
        "core",
        "inner_divertor",
        "outer_divertor",
        "pfr",
        "yguards",
]

poloidal_selections["dn"] = [
        "inner_lower_target",
        "inner_upper_target",
        "outer_lower_target",
        "outer_upper_target",
        "inner_lower_midplane",
        "inner_upper_midplane",
        "outer_lower_midplane",
        "outer_upper_midplane",
        "inner_lower_xpoint",
        "inner_upper_xpoint",
        "outer_upper_xpoint",
        "outer_lower_xpoint",
        "inner_lower_sol_extra",
        "inner_upper_sol_extra",
        "outer_upper_sol_extra",
        "outer_lower_sol_extra",
        "inner_lower_sol",
        "inner_upper_sol",
        "outer_upper_sol",
        "outer_lower_sol",
        "sol",
        "inner_core",
        "outer_core",
        "core",
        "inner_lower_divertor",
        "inner_upper_divertor",
        "outer_upper_divertor",
        "outer_lower_divertor",
        "lower_pfr",
        "upper_pfr",
        "pfr",
        "yguards",
]



plot = False
generate_data = False


def test_selectors(example_grids):

    all_grids = dict(
    sn_grids = dict(
        lsn = example_grids / "example_lsn.grd.nc",
        lsn_noguards = example_grids / "example_lsn_noguards.grd.nc",
        usn = example_grids / "example_usn.grd.nc",
        usn_noguards = example_grids / "example_usn_noguards.grd.nc",
    ),

    dn_grids = dict(
        cdn = example_grids / "example_cdn.grd.nc",
        cdn_noguards = example_grids / "example_cdn_noguards.grd.nc",
        ldn = example_grids / "example_ldn.grd.nc",
        ldn_noguards = example_grids / "example_ldn_noguards.grd.nc",
        udn2 = example_grids / "example_udn2.grd.nc",
        udn2_noguards = example_grids / "example_udn2_noguards.grd.nc",
    )
)    
    
    if plot:
        output_dir = Path("poloidal_selection_images")
        output_dir.mkdir(exist_ok=True)


    for topology_type in ["sn", "dn"]:
        print(f"Testing {topology_type} selectors...")
        grids = all_grids[f"{topology_type}_grids"]
        nrows = len(grids)

        for i, selection in enumerate(poloidal_selections[topology_type]):

            

            print(f"{selection}")

            if plot:
                fig = plt.figure(figsize=(6, nrows * 4))
                gs = fig.add_gridspec(nrows, 2, width_ratios=[1, 1], hspace=0.5, wspace=0.1)

            for row, (grid_name, path) in enumerate(grids.items()):
                if "guards" in selection and "noguards" in grid_name:
                    continue
                print(f"{grid_name} ", end="")
                ds = xhermes.HypnotoadGrid(path)

                # Generate plots for visual check
                if plot:
                    ax0 = fig.add_subplot(gs[row, 0])
                    ax1 = fig.add_subplot(gs[row, 1])
                    xhermes.plot_selection(
                        ds,
                        selection=(slice(None), xhermes.slice_poloidal(ds, selection)),
                        axes=[ax0, ax1]
                    )
                    ax0.set_title(f"{grid_name}\nLogical")
                    ax1.set_title(f"{grid_name}\nPoloidal")
                        
                # Generate test data and compare to expected
                R_test = ds["Rxy"][slice(None), xhermes.slice_poloidal(ds, selection)]

                if generate_data:
                    if selection not in Rcoords:
                        Rcoords[selection] = {}
                    Rcoords[selection][grid_name] = R_test.tolist()
                else:
                    R_expected = Rcoords[selection][grid_name]
                    npt.assert_array_equal(R_test, R_expected,
                                        err_msg=f"Selector mismatch for {selection} in {grid_name}!")

            if plot:
                fig.savefig(output_dir / f"{selection}-{topology_type}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

    if generate_data:
        with open(RCOORDS_FILE, "w") as f:
            json.dump(Rcoords, f, indent=2)
        print(f"Saved expected data to {RCOORDS_FILE}")
        
            
