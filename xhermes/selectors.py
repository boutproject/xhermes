import numpy as np


def _selector_to_indices(selector, size):
    if isinstance(selector, slice):
        start, stop, step = selector.indices(size)
        return np.arange(start, stop, step, dtype=int)

    return selector


def get_poloidal_slices(ds):
    """

    Returns poloidal indices/slices for named regions within a dataset.

    NOTES ON CONVENTION:
    - The slices will always include the first guard cell.

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

        else:
            index["inner_target"] = nyg - MYG - 1
            index["outer_target"] = MYG

            index["inner_xpoint"] = j1_1g
            index["outer_xpoint"] = j1_1g

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
            index["inner_sol_extra"] = slice(
                MYG_half, index["inner_upper_midplane"] + 1
            )
            index["outer_sol_extra"] = slice(
                index["outer_lower_midplane"] - 1, nyg - MYG_half
            )

            # SOL starting at the first cell centre after the midplane
            index["inner_sol"] = slice(MYG_half, index["inner_upper_midplane"])
            index["outer_sol"] = slice(index["outer_lower_midplane"], nyg - MYG_half)

            # SOL from target to target
            index["sol"] = slice(MYG_half, nyg - MYG_half)

            # SOL starting in the first cell centre after X-point
            index["inner_divertor"] = slice(MYG_half, j1_1g + 1)
            index["outer_divertor"] = slice(j2_2g + 1, nyg - MYG_half)

            index["pfr"] = np.r_[index["inner_divertor"], index["outer_divertor"]]

        if "upper" in topology:
            index["inner_lower_midplane"] = int(peaks[2]) - 1
            index["inner_upper_midplane"] = int(peaks[2])
            index["outer_lower_midplane"] = int(peaks[1]) + 1
            index["outer_upper_midplane"] = int(peaks[1])

            # SOL starting in cell before midplane so that you can interpolate to exact midplane
            index["inner_sol_extra"] = slice(
                index["inner_lower_midplane"], nyg - MYG_half
            )
            index["outer_sol_extra"] = slice(
                MYG_half, index["outer_lower_midplane"] + 1
            )

            # SOL starting at the first cell centre after the midplane
            index["inner_sol"] = slice(
                index["inner_lower_midplane"] + 1, nyg - MYG_half
            )
            index["outer_sol"] = slice(MYG_half, index["outer_lower_midplane"])

            # SOL from target to target
            index["sol"] = slice(MYG_half, nyg - MYG_half)

            # SOL starting in the first cell centre after X-point
            index["inner_divertor"] = slice(j2_2g + 1, nyg - MYG_half)
            index["outer_divertor"] = slice(MYG_half, j1_1g + 1)

            index["inner_pfr"] = index["inner_divertor"]
            index["outer_pfr"] = index["outer_divertor"]

            index["pfr"] = np.r_[index["inner_divertor"], index["outer_divertor"]]

    elif "double-null" in topology:
        # Targets
        index["inner_lower_target"] = MYG
        index["outer_lower_target"] = nyg - MYG - 1
        index["inner_upper_target"] = ny_innerg - MYG * 2 - 1
        index["outer_upper_target"] = ny_innerg

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
        index["inner_lower_sol_extra"] = slice(
            MYG_half, index["inner_lower_midplane"] + 2
        )
        index["inner_upper_sol_extra"] = slice(
            index["inner_upper_midplane"] - 1, ny_innerg - MYG - MYG_half
        )
        index["outer_upper_sol_extra"] = slice(
            ny_innerg - MYG_half, index["outer_lower_midplane"] + 1
        )
        index["outer_lower_sol_extra"] = slice(
            index["outer_lower_midplane"] - 1, nyg - MYG_half
        )

        # SOL starting at the first cell centre after the midplane
        index["inner_lower_sol"] = slice(MYG_half, index["inner_lower_midplane"] + 1)
        index["inner_upper_sol"] = slice(
            index["inner_upper_midplane"], ny_innerg - MYG - MYG_half
        )
        index["outer_upper_sol"] = slice(
            ny_innerg - MYG_half, index["outer_lower_midplane"]
        )
        index["outer_lower_sol"] = slice(index["outer_lower_midplane"], nyg - MYG_half)

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
        index["inner_lower_divertor"] = slice(MYG_half, j1_1g + 1)
        index["inner_upper_divertor"] = slice(j2_1g + 1, ny_innerg - MYG - MYG_half)
        index["outer_upper_divertor"] = slice(ny_innerg - MYG_half, j1_2g + 1)
        index["outer_lower_divertor"] = slice(j2_2g + 1, nyg - MYG_half)

        index["inner_sol"] = slice(MYG_half, ny_innerg - MYG - MYG_half)
        index["outer_sol"] = slice(ny_innerg - MYG_half, nyg - MYG_half)

        index["sol"] = np.r_[index["inner_sol"], index["outer_sol"]]

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


def slice_2d(ds, name, guards=False):
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

    i_pol = m["poloidal_slices"]
    if guards:
        slice_x_domain = slice(None, None)
    else:
        slice_x_domain = slice(MXG, nxg - MXG)  # Domain X points (excl guards)
    slice_x_outer = nxg - MXG - 1  # Last domain cell on SOL edge side
    slice_x_inner = MXG

    slices = {}

    slices["xguards"] = (
        np.r_[slice(0, MXG), slice(nxg - MXG, nxg)],
        slice(None, None),
    )

    ## Single null
    if "single" in topology:
        slices["targets"] = (
            slice_x_domain,
            np.r_[i_pol["inner_target"], i_pol["outer_target"]],
        )

        slices["core_boundary"] = (slice_x_inner, slice(j1_1g + 1, j2_2g + 1))

        ## Lower single null
        if "lower" in topology:
            slices["sol_boundary"] = (
                slice_x_outer,
                slice(i_pol["inner_target"], i_pol["outer_target"]),
            )
            slices["pfr_boundary"] = (
                slice_x_inner,
                np.r_[slice(MYG, j1_1g + 1), slice(j2_2g + 1, nyg - MYG)],
            )

        ## Upper single null
        elif "upper" in topology:
            slices["sol_boundary"] = (
                slice_x_outer,
                slice(i_pol["outer_target"], i_pol["inner_target"]),
            )
            slices["pfr_boundary"] = (
                slice_x_inner,
                np.r_[slice(MYG, j1_1g + 1), slice(j2_2g + 1, nyg - MYG)],
            )
        else:
            raise ValueError("Single null topology must be either upper or lower.")

    ## Double null
    elif "double" in topology:
        slices["targets"] = (
            slice_x_domain,
            np.r_[
                i_pol["inner_lower_target"],
                i_pol["inner_upper_target"],
                i_pol["outer_upper_target"],
                i_pol["outer_lower_target"],
            ],
        )

        slices["sol_inner_boundary"] = (
            slice_x_outer,
            np.r_[slice(i_pol["inner_lower_target"], i_pol["inner_upper_target"] + 1),],
        )

        slices["sol_outer_boundary"] = (
            slice_x_outer,
            np.r_[slice(i_pol["outer_upper_target"], i_pol["outer_lower_target"] + 1),],
        )

        slices["sol_boundary"] = (
            slice_x_outer,
            np.r_[
                slice(i_pol["inner_lower_target"], i_pol["inner_upper_target"] + 1),
                slice(i_pol["outer_upper_target"], i_pol["outer_lower_target"] + 1),
            ],
        )

        slices["core_boundary"] = (
            slice_x_inner,
            np.r_[slice(j1_1g + 1, j2_1g + 1), slice(j1_2g + 1, j2_2g + 1)],
        )

        slices["pfr_boundary"] = (
            slice_x_inner,
            np.r_[
                slice(MYG, j1_1g + 1),
                slice(j2_1g + 1, ny_innerg - MYG * 2),
                slice(ny_innerg, j1_2g + 1),
                slice(j2_2g + 1, nyg - MYG),
            ],
        )

    else:
        raise ValueError(f"Unknown topology: {topology}")

    # Poloidal selections
    if name in m["poloidal_slices"]:
        slices[name] = (slice_x_domain, m["poloidal_slices"][name])

    if name not in slices:
        raise ValueError(f"Unknown slice name: {name}")

    return slices[name]
