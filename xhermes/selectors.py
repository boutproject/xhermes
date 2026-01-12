import numpy as np

def slice_poloidal(ds, name):
    """
    
    Returns poloidal indices/slices for named regions within a dataset.
    
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
    j1_1g = m["jyseps1_1g"]
    j1_2g = m["jyseps1_2g"]
    j2_1g = m["jyseps2_1g"]
    j2_2g = m["jyseps2_2g"]
    MXG = m["MXG"]
    MYG = m["MYG"]
    nxg = m["nxg"]
    nyg = m["nyg"]
    ny_innerg = m["ny_innerg"]
    topology = m["topology"]
    
    
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
        index["inner_upper_target"] = ny_innerg - MYG * 2 -1 
        index["outer_upper_target"] = ny_innerg 

    # Guard selection
    if "guard" in name:
        if MYG > 0:
            if "single-null" in topology:
                index["yguards"] = np.r_[
                    slice(None,MYG),
                    slice(nyg-MYG, nyg)
                ]
            
            else:
                index["yguards"] = np.r_[
                    slice(None,MYG),
                    slice(ny_innerg - MYG * 2, ny_innerg - MYG),
                    slice(ny_innerg - MYG, ny_innerg),
                    slice(nyg-MYG, nyg)
                ]
            
        else:
            raise ValueError("Cannot select guards - no guards found in domain!")
        
    
    return index[name]
    

def slice_2d(ds, name):
    """
    
    Return a tuple of radial and poloidal indices/slices for regions within a dataset.
    
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
    j1_1g = m["jyseps1_1g"]
    j1_2g = m["jyseps1_2g"]
    j2_1g = m["jyseps2_1g"]
    j2_2g = m["jyseps2_2g"]
    MXG = m["MXG"]
    MYG = m["MYG"]
    nxg = m["nxg"]
    nyg = m["nyg"]
    ny_innerg = m["ny_innerg"]
    topology = m["topology"]
    
    polidx = lambda x: slice_poloidal(ds, x)
    slice_x_domain = slice(MXG, nxg - MXG)  # Domain X points (excl guards)
    slice_x_outer =  nxg - MXG - 1  # Last domain cell on SOL edge side
    slice_x_inner = MXG
    
    
    slices = {}
    
    slices["xguards"] = (
            np.r_[slice(0, MXG), slice(nxg - MXG, nxg)],
            slice(None, None),
        )
    
    ## Single null
    if "single" in topology:
        
        slices["targets"] = (slice_x_domain,
                             np.r_[polidx("inner_target"), 
                                   polidx("outer_target")])
        
        slices["core_boundary"] = (slice_x_inner,
                                   slice(j1_1g+1, j2_2g+1))
        
        ## Lower single null
        if "lower" in topology:
            slices["sol_boundary"] = (
                slice_x_outer,
                slice(polidx("inner_target"), polidx("outer_target"))
                )
            slices["pfr_boundary"] = (
                slice_x_inner,
                np.r_[
                    slice(MYG, j1_1g+1),
                    slice(j2_2g+1, nyg - MYG)
                ]
            )
            
        ## Upper single null
        elif "upper" in topology:
            slices["sol_boundary"] = (
                slice_x_outer,
                slice(polidx("outer_target"), polidx("inner_target"))
                )
            slices["pfr_boundary"] = (
                slice_x_inner,
                np.r_[
                    slice(MYG, j1_1g+1),
                    slice(j2_2g+1, nyg - MYG)
                ]
            )
        else:
            raise ValueError("Single null topology must be either upper or lower.")
        
    ## Double null
    elif "double" in topology:
        
        slices["targets"] = (slice_x_domain, 
                             np.r_[
                                polidx("inner_lower_target"), 
                                polidx("inner_upper_target"),
                                polidx("outer_upper_target"),
                                polidx("outer_lower_target"),
                                ])
        
        slices["sol_inner_boundary"] = (
            slice_x_outer,
            np.r_[
                slice(polidx("inner_lower_target"), polidx("inner_upper_target")+1),
            ])
        
        slices["sol_outer_boundary"] = (
            slice_x_outer,
            np.r_[
                slice(polidx("outer_upper_target"), polidx("outer_lower_target")+1),
            ])
        
        slices["sol_boundary"] = (
            slice_x_outer,
            np.r_[
                slice(polidx("inner_lower_target"), polidx("inner_upper_target")+1),
                slice(polidx("outer_upper_target"), polidx("outer_lower_target")+1),
            ])
        
        slices["core_boundary"] = (slice_x_inner,
                                   np.r_[
                                       slice(j1_1g+1, j2_1g+1),
                                       slice(j1_2g+1, j2_2g+1)]
                                   )
        
        slices["pfr_boundary"] = (slice_x_inner,
                                   np.r_[
                                       slice(MYG, j1_1g+1),
                                       slice(j2_1g+1, ny_innerg-MYG*2),
                                       slice(ny_innerg, j1_2g+1),
                                       slice(j2_2g+1, nyg-MYG)]
                                   )
        
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    # Poloidal selections
    if name in ["inner_target", "outer_target",
                "inner_lower_target", "inner_upper_target",
                "outer_lower_target", "outer_upper_target",
                "yguards"]:
        
        slices[name] = (slice(None), slice_poloidal(ds, name))
    
    return slices[name]
