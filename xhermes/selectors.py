import numpy as np


def selector_poloidal(ds, name=None, return_available=False):
    """

    Returns poloidal indices for named regions within a dataset.

    Parameters
    ----------
    ds : Dataset
        Either HypnotoadGrid or xBOUT/xHermes dataset.
    name : str, optional
        Name of the poloidal region to select. If None, no region is selected.
    return_available : bool, optional
        If True, return all available poloidal regions as a list.

    Returns
    -------
    slice or array
        Slice or array of integer indices describing the required poloidal region.
    """

    m = ds.metadata
    # TODO: use j1_1g etc from xBOUT once it's implemented there
    j1_1g = m["jyseps1_1g"]
    j1_2g = m["jyseps1_2g"]
    j2_1g = m["jyseps2_1g"]
    j2_2g = m["jyseps2_2g"]
    ixseps1g = m["ixseps1g"]
    MYG = m["MYG"]
    MYG_half = int(MYG / 2)
    nyg = m["nyg"]
    ny_innerg = m["ny_innerg"]
    topology = m["topology"]

    index = {}

    if "single-null" in topology:
        index["core"] = slice(j1_1g + 1, j2_2g + 1)

        # Targets and X-points
        if "lower" in topology:
            index["inner_target"] = MYG
            index["outer_target"] = nyg - MYG - 1

            index["inner_xpoint"] = j1_1g
            index["outer_xpoint"] = j2_2g + 1

        elif "upper" in topology:
            index["inner_target"] = nyg - MYG - 1
            index["outer_target"] = MYG

            index["inner_xpoint"] = j2_2g + 1
            index["outer_xpoint"] = j1_1g

        index["targets"] = np.r_[index["inner_target"], index["outer_target"]]

        # R is called Rxy in the grid..
        if "Rxy" in ds.keys():
            suffix = "xy"
        elif "R" in ds.keys():
            suffix = ""
        else:
            raise Exception("RZ coordinates not found in dataset")

        R_sep = ds[f"R{suffix}"][ixseps1g, :]
        R_diff = np.diff(R_sep)

        # Single null midplane found by R gradient sign change
        peaks = np.where(abs(np.diff(np.sign(R_diff))) > 0)[0] + 1

        if "lower" in topology:
            index["inner_lower_midplane"] = int(peaks[1])
            index["inner_upper_midplane"] = int(peaks[1]) + 1
            index["outer_lower_midplane"] = int(peaks[2])
            index["outer_upper_midplane"] = int(peaks[2]) - 1

            # SOL starting in cell before midplane so that you can interpolate to exact midplane
            index["inner_sol_extra"] = slice(MYG, index["inner_upper_midplane"] + 1)
            index["outer_sol_extra"] = slice(
                index["outer_lower_midplane"] - 1, nyg - MYG
            )
            index["inner_sol_extra_guards"] = slice(
                0, index["inner_upper_midplane"] + 1
            )
            index["outer_sol_extra_guards"] = slice(
                index["outer_lower_midplane"] - 1, nyg - 0
            )

            # SOL starting at the first cell centre after the midplane
            index["inner_sol"] = slice(MYG, index["inner_upper_midplane"])
            index["outer_sol"] = slice(index["outer_lower_midplane"], nyg - MYG)
            index["inner_sol_guards"] = slice(0, index["inner_upper_midplane"])
            index["outer_sol_guards"] = slice(index["outer_lower_midplane"], nyg - 0)

            # SOL from target to target
            index["sol"] = slice(MYG, nyg - MYG)
            index["sol_guards"] = slice(0, nyg - 0)

            # SOL starting in the first cell centre after X-point
            index["inner_divertor"] = slice(MYG, j1_1g + 1)
            index["outer_divertor"] = slice(j2_2g + 1, nyg - MYG)
            index["inner_divertor_guards"] = slice(0, j1_1g + 1)
            index["outer_divertor_guards"] = slice(j2_2g + 1, nyg - 0)

            index["pfr"] = np.r_[index["inner_divertor"], index["outer_divertor"]]
            index["pfr_guards"] = np.r_[
                index["inner_divertor_guards"], index["outer_divertor_guards"]
            ]

        if "upper" in topology:
            index["inner_lower_midplane"] = int(peaks[2]) - 1
            index["inner_upper_midplane"] = int(peaks[2])
            index["outer_lower_midplane"] = int(peaks[1]) + 1
            index["outer_upper_midplane"] = int(peaks[1])

            # SOL starting in cell before midplane so that you can interpolate to exact midplane
            index["inner_sol_extra"] = slice(index["inner_lower_midplane"], nyg - MYG)
            index["outer_sol_extra"] = slice(MYG, index["outer_lower_midplane"] + 1)
            index["inner_sol_extra_guards"] = slice(
                index["inner_lower_midplane"], nyg - 0
            )
            index["outer_sol_extra_guards"] = slice(
                0, index["outer_lower_midplane"] + 1
            )

            # SOL starting at the first cell centre after the midplane
            index["inner_sol"] = slice(index["inner_lower_midplane"] + 1, nyg - MYG)
            index["inner_sol_guards"] = slice(
                index["inner_lower_midplane"] + 1, nyg - 0
            )
            index["outer_sol"] = slice(MYG, index["outer_lower_midplane"])
            index["outer_sol_guards"] = slice(0, index["outer_lower_midplane"])

            # SOL from target to target
            index["sol"] = slice(MYG, nyg - MYG)
            index["sol_guards"] = slice(0, nyg - 0)

            # SOL starting in the first cell centre after X-point
            index["inner_divertor"] = slice(j2_2g + 1, nyg - MYG)
            index["outer_divertor"] = slice(MYG, j1_1g + 1)
            index["inner_divertor_guards"] = slice(j2_2g + 1, nyg - 0)
            index["outer_divertor_guards"] = slice(0, j1_1g + 1)

            index["inner_pfr"] = index["inner_divertor"]
            index["outer_pfr"] = index["outer_divertor"]
            index["inner_pfr_guards"] = index["inner_divertor_guards"]
            index["outer_pfr_guards"] = index["outer_divertor_guards"]

            index["pfr"] = np.r_[index["inner_divertor"], index["outer_divertor"]]
            index["pfr_guards"] = np.r_[
                index["inner_divertor_guards"], index["outer_divertor_guards"]
            ]

    elif "double-null" in topology:
        # Targets
        index["inner_lower_target"] = MYG
        index["outer_lower_target"] = nyg - MYG - 1
        index["inner_upper_target"] = ny_innerg - MYG * 2 - 1
        index["outer_upper_target"] = ny_innerg

        index["targets"] = np.r_[
            index["inner_lower_target"],
            index["outer_lower_target"],
            index["inner_upper_target"],
            index["outer_upper_target"],
        ]

        # X-point index defined as first point in divertor region
        index["inner_lower_xpoint"] = j1_1g
        index["inner_upper_xpoint"] = j2_1g + 1
        index["outer_upper_xpoint"] = j1_2g
        index["outer_lower_xpoint"] = j2_2g + 1

        # Double null midplane found by region cuts
        index["inner_lower_midplane"] = int((j2_1g - j1_1g) / 2) + j1_1g
        index["inner_upper_midplane"] = int((j2_1g - j1_1g) / 2) + j1_1g + 1
        index["outer_upper_midplane"] = int((j2_2g - j1_2g) / 2) + j1_2g
        index["outer_lower_midplane"] = int((j2_2g - j1_2g) / 2) + j1_2g + 1

        ## SOL

        # SOL starting in cell before midplane so that you can interpolate to exact midplane
        index["inner_lower_sol_extra"] = slice(MYG, index["inner_lower_midplane"] + 2)
        index["inner_upper_sol_extra"] = slice(
            index["inner_upper_midplane"] - 1, ny_innerg - MYG - MYG
        )
        index["outer_upper_sol_extra"] = slice(
            ny_innerg, index["outer_lower_midplane"] + 1
        )
        index["outer_lower_sol_extra"] = slice(
            index["outer_lower_midplane"] - 1, nyg - MYG
        )
        index["inner_lower_sol_extra_guards"] = slice(
            0, index["inner_lower_midplane"] + 2
        )
        index["inner_upper_sol_extra_guards"] = slice(
            index["inner_upper_midplane"] - 1, ny_innerg - MYG - 0
        )
        index["outer_upper_sol_extra_guards"] = slice(
            ny_innerg - MYG, index["outer_lower_midplane"] + 1
        )
        index["outer_lower_sol_extra_guards"] = slice(
            index["outer_lower_midplane"] - 1, nyg - 0
        )

        # SOL starting at the first cell centre after the midplane
        index["inner_lower_sol"] = slice(MYG, index["inner_lower_midplane"] + 1)
        index["inner_upper_sol"] = slice(
            index["inner_upper_midplane"], ny_innerg - MYG - MYG
        )
        index["outer_upper_sol"] = slice(ny_innerg - 0, index["outer_lower_midplane"])
        index["outer_lower_sol"] = slice(index["outer_lower_midplane"], nyg - MYG)

        index["inner_lower_sol_guards"] = slice(0, index["inner_lower_midplane"] + 1)
        index["inner_upper_sol_guards"] = slice(
            index["inner_upper_midplane"], ny_innerg - MYG - 0
        )
        index["outer_upper_sol_guards"] = slice(
            ny_innerg - MYG, index["outer_lower_midplane"]
        )
        index["outer_lower_sol_guards"] = slice(index["outer_lower_midplane"], nyg - 0)

        # SOL upstream region (midplane to X-point)
        index["inner_lower_upstream"] = slice(
            index["inner_lower_xpoint"] + 1, index["inner_lower_midplane"] + 1
        )
        index["inner_upper_upstream"] = slice(
            index["inner_upper_midplane"], index["inner_upper_xpoint"]
        )
        index["outer_upper_upstream"] = slice(
            index["outer_upper_xpoint"] + 1, index["outer_upper_midplane"] + 1
        )
        index["outer_lower_upstream"] = slice(
            index["outer_lower_midplane"], index["outer_lower_xpoint"]
        )

        # SOL upstream region (midplane to X-point) with an extra cell beyond the midplane
        index["inner_lower_upstream_extra"] = slice(
            index["inner_lower_xpoint"] + 1, index["inner_lower_midplane"] + 2
        )
        index["inner_upper_upstream_extra"] = slice(
            index["inner_upper_midplane"] - 1, index["inner_upper_xpoint"]
        )
        index["outer_upper_upstream_extra"] = slice(
            index["outer_upper_xpoint"] + 1, index["outer_upper_midplane"] + 2
        )
        index["outer_lower_upstream_extra"] = slice(
            index["outer_lower_midplane"] - 1, index["outer_lower_xpoint"]
        )

        # SOL starting at the first cell centre after the X-point
        index["inner_lower_divertor"] = slice(MYG, j1_1g + 1)
        index["inner_upper_divertor"] = slice(j2_1g + 1, ny_innerg - MYG - MYG)
        index["outer_upper_divertor"] = slice(ny_innerg, j1_2g + 1)
        index["outer_lower_divertor"] = slice(j2_2g + 1, nyg - MYG)

        index["inner_lower_divertor_guards"] = slice(0, j1_1g + 1)
        index["inner_upper_divertor_guards"] = slice(j2_1g + 1, ny_innerg - MYG - 0)
        index["outer_upper_divertor_guards"] = slice(ny_innerg - MYG, j1_2g + 1)
        index["outer_lower_divertor_guards"] = slice(j2_2g + 1, nyg - 0)

        index["inner_sol"] = slice(MYG, ny_innerg - MYG - MYG)
        index["outer_sol"] = slice(ny_innerg, nyg - MYG)

        index["sol"] = np.r_[index["inner_sol"], index["outer_sol"]]

        index["inner_sol_guards"] = slice(0, ny_innerg - MYG - 0)
        index["outer_sol_guards"] = slice(ny_innerg - MYG, nyg - 0)

        index["sol_guards"] = np.r_[
            index["inner_sol_guards"], index["outer_sol_guards"]
        ]

        # Core and PFR
        index["inner_core"] = slice(j1_1g + 1, j2_1g + 1)
        index["outer_core"] = slice(j1_2g + 1, j2_2g + 1)
        index["core"] = np.r_[index["inner_core"], index["outer_core"]]

        index["inner_lower_pfr"] = index["inner_lower_divertor"]
        index["inner_upper_pfr"] = index["inner_upper_divertor"]
        index["outer_lower_pfr"] = index["outer_lower_divertor"]
        index["outer_upper_pfr"] = index["outer_upper_divertor"]

        index["lower_pfr"] = np.r_[
            index["inner_lower_divertor"], index["outer_lower_divertor"]
        ]
        index["upper_pfr"] = np.r_[
            index["outer_upper_divertor"], index["inner_upper_divertor"]
        ]
        index["pfr"] = np.r_[index["lower_pfr"], index["upper_pfr"]]

        index["inner_lower_pfr_guards"] = index["inner_lower_divertor_guards"]
        index["inner_upper_pfr_guards"] = index["inner_upper_divertor_guards"]
        index["outer_lower_pfr_guards"] = index["outer_lower_divertor_guards"]
        index["outer_upper_pfr_guards"] = index["outer_upper_divertor_guards"]

        index["lower_pfr_guards"] = np.r_[
            index["inner_lower_divertor_guards"], index["outer_lower_divertor_guards"]
        ]
        index["upper_pfr_guards"] = np.r_[
            index["outer_upper_divertor_guards"], index["inner_upper_divertor_guards"]
        ]
        index["pfr_guards"] = np.r_[
            index["lower_pfr_guards"], index["upper_pfr_guards"]
        ]

    # Guard selection
    if MYG > 0:
        if "single-null" in topology:
            index["yguards"] = np.r_[slice(None, MYG), slice(nyg - MYG, nyg)]

        else:
            index["yguards"] = np.r_[
                slice(None, MYG),
                slice(ny_innerg - MYG * 2, ny_innerg - MYG),
                slice(ny_innerg - MYG, ny_innerg),
                slice(nyg - MYG, nyg),
            ]
    if not return_available and not name:
        raise ValueError(
            "Must specify a poloidal region to select, or set return_available to True"
        )
    if return_available:
        return list(index.keys())

    if name in index:
        return index[name]
    else:
        raise ValueError(
            f"Unknown poloidal selection {name}. Must be one of {list(index.keys())}"
        )


def selector_radial(ds, radial_region=None, return_available=False):
    """

    Return a slice of radial indices for a given radial region.

    Parameters
    ----------
    ds : Dataset
        Either HypnotoadGrid or xBOUT/xHermes dataset.
    radial_region : str, optional
        Name of the radial region to select (e.g., "domain", "domain_guards", "xlow_boundary",
        "xlow_boundary_guard", "xup_boundary", "xup_boundary_guard").
    return_available : bool, optional
        If True, return all available radial regions as a list.

    Returns
    -------
    slice
        Slice describing the required radial region within the dataset.

    """

    m = ds.metadata

    index = {
        "domain": slice(m["MXG"], m["nxg"] - m["MXG"]),
        "domain_guards": slice(0, m["nxg"]),
        "xguards": np.r_[slice(0, m["MXG"]), slice(m["nxg"] - m["MXG"], m["nxg"])],
        "boundary_xlow": m["MXG"],
        "boundary_guard_xlow": m["MXG"] - 1,
        "boundary_xup": m["nxg"] - m["MXG"] - 1,
        "boundary_guard_xup": m["nxg"] - m["MXG"],
        "sol": slice(m["ixseps1g"], m["nxg"] - m["MXG"]),
        "sol_guards": slice(m["ixseps1g"], m["nxg"]),
        "core": slice(m["MXG"], m["ixseps1g"]),
        "core_guards": slice(0, m["ixseps1g"]),
    }

    index["pfr"] = index["core"]
    index["pfr_guards"] = index["core_guards"]

    if return_available:
        return list(index.keys())

    if radial_region is None:
        raise ValueError(
            "Must specify a radial region to select, or set return_available to True"
        )

    if "guard" in radial_region and m["MXG"] == 0:
        raise ValueError("Cannot select X guard cells if they don't exist! MXG is 0")

    if radial_region in index:
        return index[radial_region]

    raise ValueError(
        f"Unknown radial region {radial_region}. Must be one of {list(index.keys())}"
    )


def selector_2d(ds, radial_region, poloidal_region):
    """

    Combines radial and poloidal selection with special handling of incompatible combinations (e.g., SOL and inner boundary).
    Adds "boundary" and "boundary_guard" option for radial_region which picks the side automatically.
    Returns a tuple of radial and poloidal slices.

    Parameters
    ----------
    ds : xarray.Dataset-like
        Either HypnotoadGrid or xBOUT/xHermes dataset.
    radial_region : str
        Name of the radial region to select (e.g., "domain", "domain_guards", "boundary", "boundary_guard").
    poloidal_region : str
        Name of the poloidal region to select (see xhermes.selectors.selector_poloidal for options).

    Returns
    -------
    tuple
        Tuple of slices defining the requested region within the dataset.

    """

    m = ds.metadata

    ## Handle selecting guard cells if no guards present
    if "guard" in radial_region and m["MXG"] == 0:
        raise ValueError("Cannot select X guard cells if they don't exist! MXG is 0")
    if "guard" in poloidal_region and m["MYG"] == 0:
        raise ValueError("Cannot select Y guard cells if they don't exist! MYG is 0")

    ## When radial_region is just "boundary", automatically pick the side
    if radial_region == "boundary" or radial_region == "boundary_guard":
        if any([x in poloidal_region for x in ["sol", "divertor"]]):
            x_side = "xup"
        elif any([x in poloidal_region for x in ["core", "pfr"]]):
            x_side = "xlow"
        else:
            raise ValueError(
                f"Cannot automatically determine which boundary to select for poloidal region `{poloidal_region}`. "
                "Please specify the radial region as either 'boundary_xlow', 'boundary_guard_xlow', 'boundary_xup', or 'boundary_guard_xup'."
            )
        radial_region = f"{radial_region}_{x_side}"

    ## Handle selecting wrong radial region side
    if radial_region == "boundary_xlow" and any(
        [x in poloidal_region for x in ["sol", "divertor"]]
    ):
        raise ValueError(
            f"Cannot select {poloidal_region} with {radial_region}: "
            "SOL and divertor regions don't extend to the inner boundary."
        )
    if radial_region == "boundary_xup" and any(
        [x in poloidal_region for x in ["core", "pfr"]]
    ):
        raise ValueError(
            f"Cannot select {poloidal_region} with {radial_region}: "
            "Core and PFR regions don't extend to the outer boundary."
        )

    radial_selection = selector_radial(ds, radial_region)
    poloidal_selection = selector_poloidal(ds, poloidal_region)

    if (
        isinstance(radial_selection, np.ndarray)
        and isinstance(poloidal_selection, np.ndarray)
        and radial_selection.size > 1
        and poloidal_selection.size > 1
    ):
        message = (
            "Cannot combine radial and poloidal selections when both resolve to "
            "non-contiguous index arrays. This produces unsupported paired advanced "
            "indexing rather than a 2D region. Use two-pass handling instead, for "
            "example plot the X and Y selections separately or clear guards in two passes."
        )
        print(message)
        raise ValueError(message)

    return (radial_selection, poloidal_selection)


def _select_region(ds, radial_region=None, poloidal_region=None, custom_selection=None):
    """
    Select a radial/poloidal region from the a Dataset or DataArray

    Parameters
    ----------
    poloidal_region : str
        Poloidal region name to select. See xhermes.selectors.selector_poloidal.
    radial_region : str
        Radial region name to select. See xhermes.selectors.selector_2d.
    custom_selection : tuple of slices, optional
        Custom selection in the form (slice(x_start, x_end), slice(theta_start, theta_end)).
        If provided, this will override the radial_region and poloidal_region parameters.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Dataset with data selected for the specified region.
    """
    if custom_selection is not None:
        selection = custom_selection
    else:
        selection = selector_2d(ds, radial_region, poloidal_region)

    return ds.isel(x=selection[0], theta=selection[1])
