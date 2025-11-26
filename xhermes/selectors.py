import numpy as np

def select_poloidal(ds, name):
    
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
    
    index = {}
    
    if "single-null" in topology:
        if any([x in name for x in ["inner_lower", "inner_upper", "outer_lower", "outer_upper"]]):
            raise ValueError(f"{name} region not present in {topology}")
        
        # if 
        # index["inner_target"] = MYG
        # index["outer_target"] = nyg - MYG
    
    return index[name]
    

def select_region(ds, name):
    
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
    
    if "single-null" in topology:
        
        slices["inner_target_guards"] = ()
    
    return slices[name]

    slices["inner_lower_target_guards"] = (target_xslice, slice(0, MYG))
    slices["inner_upper_target_guards"] = (
        target_xslice,
        slice(ny_inner + MYG, ny_inner + MYG * 2),
    )
    slices["outer_upper_target_guards"] = (
        target_xslice,
        slice(ny_inner + MYG * 2, ny_inner + MYG * 3),
    )
    slices["outer_lower_target_guards"] = (
        target_xslice,
        slice(nyg - MYG, nyg),
    )