import numpy as np


def _selector_to_indices(selector, size):
    if isinstance(selector, slice):
        start, stop, step = selector.indices(size)
        return np.arange(start, stop, step, dtype=int)

    return selector


def get_poloidal_selections(ds):
    """

    Returns poloidal indices for named regions within a dataset.

    Parameters
    ----------
    ds : Dataset
        Either HypnotoadGrid or xBOUT/xHermes dataset.

    Returns
    -------
    dict
        Dictionary containing integer indices or integer arrays describing the
        required poloidal regions.
    """

    m = ds.metadata
    # TODO: use j1_1g etc from xBOUT once it's implemented there
    j1_1g = m["jyseps1_1g"]
    j1_2g = m["jyseps1_2g"]
    j2_1g = m["jyseps2_1g"]
    j2_2g = m["jyseps2_2g"]
    ixseps1 = m["ixseps1"]
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

        R_sep = ds[f"R{suffix}"][ixseps1, :]
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

    return {
        name: _selector_to_indices(selector, nyg) for name, selector in index.items()
    }


def select_2d(ds, poloidal_selection, radial_selection):
    """

    Return a tuple of radial and poloidal indices/slices for regions within a dataset.

    Parameters
    ----------
    ds : xarray.Dataset-like
        Either HypnotoadGrid or xBOUT/xHermes dataset.
    poloidal_selection : str
        Name of the poloidal region to select (see xhermes.selectors.get_poloidal_selections for options).
    radial_selection : str
        Name of the radial region to select (e.g., "domain", "domain_guards", "inner_boundary",
        "inner_guard", "outer_boundary", "outer_guard").

    Returns
    -------
    tuple
        Tuple of slices defining the requested region within the dataset.

    """

    m = ds.metadata

    if "guard" in radial_selection and m["MXG"] == 0:
        raise ValueError("Cannot select X guard cells if they don't exist! MXG is 0")
    if "guard" in poloidal_selection and m["MYG"] == 0:
        raise ValueError("Cannot select Y guard cells if they don't exist! MYG is 0")

    if radial_selection == "domain":
        x = slice(m["MXG"], m["nxg"] - m["MXG"])
    elif radial_selection == "domain_guards":
        x = slice(0, m["nxg"])
    elif radial_selection == "inner_boundary":
        x = slice(m["MXG"], m["MXG"] + 1)
    elif radial_selection == "inner_boundary_guard":
        x = slice(m["MXG"] - 1, m["MXG"])
    elif radial_selection == "outer_boundary":
        x = slice(m["nxg"] - m["MXG"] - 1, m["nxg"] - m["MXG"])
    elif radial_selection == "outer_boundary_guard":
        x = slice(m["nxg"] - m["MXG"], m["nxg"] - m["MXG"] + 1)
    else:
        raise ValueError(
            f'Unknown radial selection {radial_selection}. Must be one of "domain", "domain_guards", "inner_boundary", '
            f'"inner_guard", "outer_boundary", "outer_guard"'
        )

    ## Handle special cases
    if radial_selection == "inner_boundary" and any(
        [x in poloidal_selection for x in ["sol", "divertor"]]
    ):
        raise ValueError(
            f"Cannot select {poloidal_selection} with {radial_selection}: "
            "SOL and divertor regions don't extend to the inner boundary."
        )

    y = m["poloidal_slices"][poloidal_selection]

    return (x, y)
