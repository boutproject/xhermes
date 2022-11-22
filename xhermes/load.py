from xbout import open_boutdataset

import os


def open_hermesdataset(
    datapath="./BOUT.dmp.*.nc",
    inputfilepath=None,
    chunks={},
    run_name=None,
    info=True,
    unnormalise=True,
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
        info=False,
        **kwargs,
    )

    # Normalisation
    meta = ds.attrs["metadata"]
    Cs0 = meta["Cs0"]
    Omega_ci = meta["Omega_ci"]
    rho_s0 = meta["rho_s0"]
    Nnorm = meta["Nnorm"]
    Tnorm = meta["Tnorm"]

    # SI values
    Mp = 1.67e-27  # Proton mass
    e = 1.602e-19  # Coulombs

    # Coordinates
    ds.t.attrs["units_type"] = "hermes"
    ds.t.attrs["units"] = "s"
    ds.t.attrs["conversion"] = 1.0 / Omega_ci

    for varname in ds:
        da = ds[varname]
        if len(da.dims) == 4:
            # 4D field => Mark as Hermes-normalised data
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
    where 'data' is a directory

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
