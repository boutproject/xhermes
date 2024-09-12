import xarray

def guard_replace_1d(da):
    """
    Replace the inner guard cells with the values of their respective
    cell edges, i.e. the values at the model inlet and at the target.
    This is done by interpolating the value between the two neighbouring
    cell centres. Checks for presence of guard cells if passed a DataArray.

    Cell order at target:
    ... | last | guard | second guard (unused)
                ^target      
        |  -3  |  -2   |      -1
        
    Parameters
    ----------
    - da: Numpy array or Xarray DataArray with guard cells
        
    Returns
    ----------
    - Numpy array with guard replacement

    """

    da = da.copy()
    
    if type(da) == xarray.core.dataarray.DataArray:
        if da.metadata["keep_yboundaries"] is False:
            raise ValueError("Cannot guard replace if y-boundaries are not kept")
        
        if da.name in da.coords:
            raise ValueError("Cannot guard replace DataArray if it is a coordinate, try passing da.values() instead")
        
        da[{"pos" : -2}] = (da[{"pos" : -2}] + da[{"pos" : -3}])/2
        da[{"pos" : 1}] = (da[{"pos" : 1}] + da[{"pos" : 2}])/2
        
        da = da.isel(pos = slice(1, -1))
        
    else:
        da[-2] = (da[-2] + da[-3])/2
        da[1] = (da[1] + da[2])/2
        da = da[1:-1]

    return da