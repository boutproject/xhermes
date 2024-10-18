from xbout import open_boutdataset

import os


def open_hermesdataset(
    datapath="./BOUT.dmp.*.nc",
    inputfilepath=None,
    chunks={},
    run_name=None,
    info=False,
    unnormalise=True,
    debug_variable_names = False,
    **kwargs,
):
    """
    Load a dataset from a set of Hermes output files

    Example:
    ```
    import xhermes
    ds = xhermes.open_hermesdataset('data/BOUT.dmp.*.nc')
    ```

    Returns
    -------

    ds : xarray.Dataset
    """

    ds = open_boutdataset(
        datapath=datapath,
        inputfilepath=inputfilepath,
        chunks=chunks,
        run_name=run_name,
        info=info,
        **kwargs,
    )
    
    # Record if guard cells read in or not
    if "keep_xboundaries" in kwargs:
        ds.metadata["keep_xboundaries"] = kwargs["keep_xboundaries"]
    else:
        ds.metadata["keep_xboundaries"] = True   # Default
        
    if "keep_yboundaries" in kwargs:
        ds.metadata["keep_yboundaries"] = kwargs["keep_yboundaries"]
    else:
        ds.metadata["keep_yboundaries"] = False   # Default

    # Normalisation
    meta = ds.attrs["metadata"]
    Cs0 = meta["Cs0"]
    Omega_ci = meta["Omega_ci"]
    rho_s0 = meta["rho_s0"]
    Nnorm = meta["Nnorm"]
    Tnorm = meta["Tnorm"]
    Bnorm = meta["Bnorm"]

    # SI values
    Mp = 1.67e-27  # Proton mass
    e = 1.602e-19  # Coulombs



    for varname in list(ds.data_vars) + list(ds.coords):
        da = ds[varname]
        if len(da.dims) == 4:  # Time-evolving field
            da.attrs["units_type"] = "hermes"

            # Check if data already has units and conversion attributes
            if ("units" in da.attrs) and ("conversion" in da.attrs):
                if debug_variable_names is True:
                    print(varname + " already annotated")
                continue  # No need to add attributes

            # Mark as Hermes-normalised data
            da.attrs["units_type"] = "hermes"

            if varname[:2] == "NV":
                # Momentum
                da.attrs.update(
                    {
                        "units": "kg m / s",
                        "conversion": Mp * Nnorm * Cs0,
                        "standard_name": "momentum",
                        "long_name": varname[2:] + " parallel momentum",
                    }
                )
            elif varname[0] == "N":
                # Density
                da.attrs.update(
                    {
                        "units": "m^-3",
                        "conversion": Nnorm,
                        "standard_name": "density",
                        "long_name": varname[1:] + " number density",
                    }
                )

            elif varname[0] == "T":
                # Temperature
                da.attrs.update(
                    {
                        "units": "eV",
                        "conversion": Tnorm,
                        "standard_name": "temperature",
                        "long_name": varname[1:] + " temperature",
                    }
                )

            elif varname[0] == "V":
                # Velocity
                da.attrs.update(
                    {
                        "units": "m / s",
                        "conversion": Cs0,
                        "standard_name": "velocity",
                        "long_name": varname[1:] + " parallel velocity",
                    }
                )

            elif varname[0] == "P":
                # Pressure
                da.attrs.update(
                    {
                        "units": "Pa",
                        "conversion": e * Tnorm * Nnorm,
                        "standard_name": "pressure",
                        "long_name": varname[1:] + " pressure",
                    }
                )
            elif varname == "phi":
                # Potential
                da.attrs.update(
                    {
                        "units": "V",
                        "conversion": Tnorm,
                        "standard_name": "potential",
                        "long_name": "Plasma potential",
                    }
                )
            else:
                # Don't know what this is
                da.attrs["units_type"] = "unknown"

                
                
        # Coordinates
        if varname == "dy":
            # Poloidal cell angular width
            da.attrs.update(
                {
                    "units": "radian",
                    "conversion": 1,
                    "standard_name": "poloidal cell angular width",
                    "long_name": "Poloidal cell angular width",
                }
            )
        elif varname == "dx":
            # Radial cell width
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "Wb",
                    "conversion": rho_s0**2 * Bnorm,
                    "standard_name": "radial cell width",
                    "long_name": "Radial cell width in flux space",
                }
            )
        elif varname == "dz":
            # Radial cell width
            da.attrs.update(
                {
                    "units_type": "SI",
                    "units": "radian",
                    "conversion": 1,
                    "standard_name": "toroidal cell angular width",
                    "long_name": "Poloidal cell angular width",
                }
            )
        elif varname == "J":
            # Jacobian
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "m/radian T",
                    "conversion": rho_s0 / Bnorm,
                    "standard_name": "Jacobian",
                    "long_name": "Jacobian to translate from flux to cylindrical coordinates in real space",
                }
            )
        elif varname == "g11":
            # Metric tensor term
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "T2 m2",
                    "conversion": (Bnorm * rho_s0)**2,
                    "standard_name": "g11",
                    "long_name": "g11 term in the metric tensor",
                }
            )
        elif varname == "g22":
            # Metric tensor term
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "m-2",
                    "conversion": 1/(rho_s0)**2,
                    "standard_name": "g22",
                    "long_name": "g22 term in the metric tensor",
                }
            )
        elif varname == "g33":
            # Metric tensor term
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "m-2",
                    "conversion": 1/(rho_s0)**2,
                    "standard_name": "g33",
                    "long_name": "g33 term in the metric tensor",
                }
            )
        elif varname == "g12":
            # Metric tensor term
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "T",
                    "conversion": Bnorm,
                    "standard_name": "g12",
                    "long_name": "g12 term in the metric tensor",
                }
            )
        elif varname == "g13":
            # Metric tensor term
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "T",
                    "conversion": Bnorm,
                    "standard_name": "g13",
                    "long_name": "g13 term in the metric tensor",
                }
            )
        elif varname == "g23":
            # Metric tensor term
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "m-2",
                    "conversion": 1/(rho_s0)**2,
                    "standard_name": "g23",
                    "long_name": "g23 term in the metric tensor",
                }
            )
        elif varname == "g_11":
            # Metric tensor term
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "T-2m-2",
                    "conversion": 1/(Bnorm * rho_s0)**2,
                    "standard_name": "g_11",
                    "long_name": "g_11 term in the metric tensor",
                }
            )
        elif varname == "g_22":
            # Metric tensor term
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "m2",
                    "conversion": (rho_s0)**2,
                    "standard_name": "g_22",
                    "long_name": "g_22 term in the metric tensor",
                }
            )
        elif varname == "g_33":
            # Metric tensor term
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "m2",
                    "conversion": (rho_s0)**2,
                    "standard_name": "g_33",
                    "long_name": "g_33 term in the metric tensor",
                }
            )
        elif varname == "g_12":
            # Metric tensor term
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "T-1",
                    "conversion": 1/Bnorm,
                    "standard_name": "g_12",
                    "long_name": "g_12 term in the metric tensor",
                }
            )
        elif varname == "g_13":
            # Metric tensor term
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "T-1",
                    "conversion": 1/Bnorm,
                    "standard_name": "g_13",
                    "long_name": "g_13 term in the metric tensor",
                }
            )
        elif varname == "g_23":
            # Metric tensor term
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "m2",
                    "conversion": (rho_s0)**2,
                    "standard_name": "g_23",
                    "long_name": "g_23 term in the metric tensor",
                }
            )
        elif varname == "t" or varname == "t_array":
            # Time
            da.attrs.update(
                {
                    "units_type": "hermes",
                    "units": "s",
                    "conversion": 1/Omega_ci,
                    "standard_name": "time",
                    "long_name": "Time",
                }
            )

    if unnormalise:
        ds.hermes.unnormalise()

    if ds.attrs["options"] is not None:
        # Process options
        options = ds.attrs["options"]

        # List of components
        component_list = [
            c.strip() for c in options["hermes"]["components"].strip(" ()\t").split(",")
        ]

        # Turn into a dictionary
        components = {}
        for component in component_list:
            if component in options:
                c_opts = options[component]
                if "type" in c_opts:
                    c_type = c_opts["type"]
                else:
                    c_type = component  # Type is the name of the component
            else:
                c_opts = None
                c_type = component
            components[component] = {"type": c_type, "options": c_opts}
        ds.attrs["components"] = components
        
        # Identify dimensions
        dims = list(ds.squeeze().dims)
        if "t" in dims: dims.remove("t")
        meta["dimensions"] = len(dims)
        
        # Identify species
        meta["species"] = [x.split("P")[1] for x in ds.data_vars if x.startswith("P") and len(x) < 4]
        meta["charged_species"] = [x for x in meta["species"] if "e" in x or "+" in x]
        meta["ion_species"] = [x for x in meta["species"] if "+" in x]
        meta["neutral_species"] = list(set(meta["species"]).difference(set(meta["charged_species"])))
        
        # Prepare dictionary mapping recycling species pairs
        if "recycling" in ds.attrs["components"]:
            meta["recycling_pairs"] = dict()
            for ion in meta["ion_species"]:
                if "recycle_as" in ds.options[ion].keys():
                    meta["recycling_pairs"][ion] = ds.options[ion]["recycle_as"]
                else:
                    print(f"No recycling partner found for {ion}")

    return ds


def open(
    path,
    **kwargs,
):
    """
    Open a data directory containing Hermes output files.

    Example:
    ```
    import xhermes
    ds = xhermes.open('data')
    ```
    where 'data' is a directory.

    Can also load a grid file containing geometry information:

    Example:
    ```
    bd = xhermes.open(".", geometry="toroidal", gridfilepath="../tokamak.nc")
    ```

    Parameters
    ----------

    path : string
        A directory containing BOUT.dmp.* files and a BOUT.settings file

    All other parameters are passed through to `open_hermesdataset`

    Returns
    -------

    ds : xarray.Dataset
    """
    return open_hermesdataset(
        datapath=os.path.join(path, "BOUT.dmp.*.nc"),
        inputfilepath=os.path.join(path, "BOUT.settings"),
        **kwargs,
    )
