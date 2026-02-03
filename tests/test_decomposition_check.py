#!/usr/bin/env python3

import xhermes
import pytest
import numpy as np
import numpy.testing as npt

from urllib.request import urlretrieve
import zipfile


expected_ncores = {
    "cdn" : [6, 12], 
    "cdn_noguards" : [6, 12], 
    "ldn" : [6, 12], 
    "ldn_noguards" : [6, 12], 
    "lsn" : [4, 8], 
    "lsn_noguards" : [4, 8], 
    "usn" : [4, 8], 
    "usn_noguards" : [4, 8], 
    "udn2" : [6, 12], 
    "udn2_noguards" : [6, 12],
}

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
    
    for geometry, expected in expected_ncores.items():
        
        ds = xhermes.HypnotoadGrid(grids[geometry])
        test = ds.num_processors(1, 12)
        
        npt.assert_array_equal(test, expected, 
                                    err_msg=f"Incorrect number of cores found for {geometry}!")
            
