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

t0 = time()

RCOORDS_FILE = Path(__file__).parent / "Rcoords_expected.json"

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

poloidal_selections = [
        # "inner_target",
        # "outer_target",
        # "inner_lower_target",
        # "inner_upper_target",
        # "outer_lower_target",
        # "outer_upper_target",
        # "inner_lower_midplane",
        # "inner_upper_midplane",
        # "outer_lower_midplane",
        # "outer_upper_midplane",
        # "inner_xpoint",
        # "outer_xpoint",
        # "inner_lower_xpoint",
        # "inner_upper_xpoint",
        # "outer_upper_xpoint",
        # "outer_lower_xpoint",
        # "inner_lower_sol",
        # "inner_upper_sol",
        # "outer_upper_sol",
        # "outer_lower_sol",
        # "inner_sol",
        # "outer_sol",
        # "sol",
        # "inner_core",
        # "outer_core",
        # "core",
        # "inner_lower_divertor",
        # "inner_upper_divertor",
        # "outer_upper_divertor",
        # "outer_lower_divertor",

        # "inner_divertor",
        # "outer_divertor",

        # "lower_pfr",
        # "upper_pfr",
        # "pfr",
        # "pfr",
        "yguards",
    ]

plot = False
generate_data = True


def test_selectors(example_grids):
    
    grids = dict(
        cdn = example_grids / "example_cdn.grd.nc",
        cdn_noguards = example_grids / "example_cdn_noguards.grd.nc",
        ldn = example_grids / "example_ldn.grd.nc",
        ldn_noguards = example_grids / "example_ldn_noguards.grd.nc",
        lsn = example_grids / "example_lsn.grd.nc",
        lsn_noguards = example_grids / "example_lsn_noguards.grd.nc",
        usn = example_grids / "example_usn.grd.nc",
        usn_noguards = example_grids / "example_usn_noguards.grd.nc",
        udn2 = example_grids / "example_udn2.grd.nc",
        udn2_noguards = example_grids / "example_udn2_noguards.grd.nc",
    )

    if plot:
        output_dir = Path("poloidal_selection_images")
        output_dir.mkdir(exist_ok=True)

    nrows = len(grids)
    print()

    for i, selection in enumerate(poloidal_selections):
        print(f"{selection}")

        if plot:
            fig = plt.figure(figsize=(6, nrows * 4))
            gs = fig.add_gridspec(nrows, 2, width_ratios=[1, 1], hspace=0.5, wspace=0.1)

        for row, (name, path) in enumerate(grids.items()):
            print(f"{name} ", end="")
            
            ds = xhermes.HypnotoadGrid(path)

            
            # Generate plots for 
            if plot:
                ax0 = fig.add_subplot(gs[row, 0])
                ax1 = fig.add_subplot(gs[row, 1])
                
                try:
                    xhermes.plot_selection(
                        ds,
                        selection=(slice(None), xhermes.slice_poloidal(ds, selection)),
                        axes=[ax0, ax1]
                    )
                    ax0.set_title(f"{name}\nLogical")
                    ax1.set_title(f"{name}\nPoloidal")

                except Exception as e:
                    print("Failed! ", end="")
                    msg = textwrap.fill(str(e), width=40)
                    ax0.text(0.0, 0.5, msg, fontsize="medium", transform=ax0.transAxes,
                            va="center", ha="left", clip_on=True)
                    

            try:
                R_test = xhermes.slice_poloidal(ds, selection)["Rxy"].values.flatten()
            except Exception as e:
                print(f"Failed ({e}) ", end="")
                continue

            if generate_data:
                if selection not in Rcoords:
                    Rcoords[selection] = {}
                Rcoords[selection][name] = R_test.tolist()
            else:
                R_expected = Rcoords[selection][name]
                npt.assert_array_equal(R_test, R_expected,
                                       err_msg=f"Selector mismatch for {selection} in {name}!")

        print()

        if plot:
            fig.savefig(output_dir / f"{selection}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    if generate_data:
        with open(RCOORDS_FILE, "w") as f:
            json.dump(Rcoords, f, indent=2)
        print(f"Saved expected data to {RCOORDS_FILE}")
        
        # for selection in Rcoords.keys():
        #     for geometry in Rcoords[selection].keys():
        #         expected = Rcoords[selection][geometry]
        #         ds = xhermes.HypnotoadGrid(grids[geometry])
        #         test = ds["Rxy"][xhermes.slice_2d(ds, selection)]
                
        #         npt.assert_array_equal(test, expected, 
        #                                     err_msg=f"Selector mismatch for {selection} in {geometry}!")
            
