import numpy as np

def poloidal_selector(ds, name):
    """
    Selects indices for requested poloidal regions.
    
    Parameters
    ----------
    ds : Dataset
        Either HypnotoadGrid or xBOUT/xHermes dataset.
    name : str
        Identifier of the poloidal region to select.

    Returns
    -------
    int or slice
        integer index or slice describing the required poloidal indices.
    """
    
    
    m = ds.metadata
    # TODO: use j1_1g etc from xBOUT once it's implemented there
    j1_1g = m["j1_1g"]
    j1_2g = m["j1_2g"]
    j2_1g = m["j2_1g"]
    j2_2g = m["j2_2g"]
    MXG = m["MXG"]
    MYG = m["MYG"]
    nxg = m["nxg"]
    nyg = m["nyg"]
    ny_inner = m["ny_inner"]
    topology = m["topology"]
    
    # Inverse of MYG needed for some double null indexing
    if MYG == 0:
        MYG_inverse = 2
    else:
        MYG_inverse = 0
    
    index = {}
    
    # Target selection
    if "single-null" in topology:
        if any([x in name for x in ["inner_lower", "inner_upper", "outer_lower", "outer_upper"]]):
            raise ValueError(f"{name} region not present in {topology}")
        
    if "single-null" in topology:
        if "lower" in topology:
            index["inner_target"] = MYG
            index["outer_target"] = nyg - MYG - 1
        else:
            index["inner_target"] = nyg - MYG - 1
            index["outer_target"] = MYG
            
    elif "double-null" in topology:
        index["inner_lower_target"] = MYG
        index["outer_lower_target"] = nyg - MYG - 1
        index["inner_upper_target"] = ny_inner - MYG_inverse + 1
        index["outer_upper_target"] = ny_inner + MYG * 3

    # Guard selection
    if MYG > 0:
        if "single-null" in topology:
            index["yguards"] = np.r_[
                slice(None,MYG),
                slice(nyg-MYG, nyg)
            ]
        
        else:
            index["yguards"] = np.r_[
                slice(None,MYG),
                slice(ny_inner+MYG, ny_inner + MYG*2),
                slice(ny_inner+MYG*2, ny_inner + MYG*3),
                slice(nyg-MYG, nyg)
            ]
           
    else:
        raise ValueError("Cannot select guards - no guards found in domain!")
        
    
    return index[name]
    

def region_selector(ds, name):
    """Select pre-defined regions within a dataset.
    
    Parameters
    ----------
    ds : xarray.Dataset-like
        Either HypnotoadGrid or xBOUT/xHermes dataset.
    name : str
        Name of the region to select (e.g., "inner_target", "yguards").
        
    Returns
    -------
    tuple
        Tuple of slices defining the requested region within the dataset.
    
    """

    
    m = ds.metadata
    # TODO: use j1_1g etc from xBOUT once it's implemented there
    j1_1g = m["j1_1g"]
    j1_2g = m["j1_2g"]
    j2_1g = m["j2_1g"]
    j2_2g = m["j2_2g"]
    MXG = m["MXG"]
    MYG = m["MYG"]
    nxg = m["nxg"]
    nyg = m["nyg"]
    ny_inner = m["ny_inner"]
    topology = m["topology"]
    
    
    slices = {}
    
    slices["xguards"] = (
            np.r_[slice(0, MXG), slice(nxg - MXG, nxg)],
            slice(None, None),
        )
    
    # Poloidal selections
    if name in ["inner_target", "outer_target",
                "inner_lower_target", "inner_upper_target",
                "outer_lower_target", "outer_upper_target",
                "yguards"]:
        
        slices[name] = (slice(None), poloidal_selector(ds, name))
    
    return slices[name]
