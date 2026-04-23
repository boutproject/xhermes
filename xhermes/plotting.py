import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .selectors import slice_2d


def plot_selection(
    ds,
    poloidal_region=None,
    radial_region=None,
    custom_selection=None,
    dpi=150,
    title="",
    axes=None,
):
    """
    Visualises selected grid region over a logical and poloidal grid plot.
    Takes the name of the poloidal and radial regions and passes them to slice_2d.
    Alternatively, can use a custom selection tuple compatible with NumPy 2D indexing.

    Parameters
    ds : dict-like
        Either a Hermes-3 results dataset or a HypnotoadGrid object. Needs to have
        metadata with region boundaries, and Rxy, Zxy and their corner coordinates
        as keys.
    poloidal_region : str, optional
        Name of the poloidal region to select (see xhermes.selectors.get_poloidal_slices for options).
        If None, please pass a custom_selection, which should be a tuple of (radial_sel, poloidal_sel)
        compatible with NumPy 2D indexing.
    radial_region : str, optional
        Name of the radial region to select (e.g., "domain", "domain_guards", "inner_boundary",
        "inner_guard", "outer_boundary", "outer_guard").
        If None, please pass a custom_selection, which should be a tuple of (radial_sel, poloidal_sel)
        compatible with NumPy 2D indexing.
    custom_selection : tuple of (radial_sel, poloidal_sel), optional
        If poloidal_region or radial_region are None, you can pass a custom selection tuple compatible
        with NumPy 2D indexing.
    dpi : int, optional
        Dots per inch for the figure. Higher values give better quality but larger file size.
    title : str, optional
        Title for the figure. Default is an empty string.
    """

    if custom_selection is None:
        if poloidal_region is None and radial_region is None:
            raise ValueError(
                "Must provide both poloidal_region and radial_region, or pass custom_selection."
            )
        if poloidal_region is None or radial_region is None:
            raise ValueError(
                "poloidal_region and radial_region must be provided together."
            )
    else:
        if poloidal_region is not None or radial_region is not None:
            raise ValueError(
                "Pass either poloidal_region/radial_region or custom_selection, not both."
            )
        if not isinstance(custom_selection, (tuple, list)) or len(custom_selection) != 2:
            raise ValueError(
                "custom_selection must be a 2-item tuple/list of (radial_sel, poloidal_sel)."
            )

    if custom_selection is not None:
        selection = custom_selection
    else:
        selection = slice_2d(ds, poloidal_region, radial_region)

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 6), dpi=dpi)
        own_fig = True
    else:
        if len(axes) != 2 or any(not hasattr(axis, "get_figure") for axis in axes):
            raise ValueError("axes must contain exactly two matplotlib Axes objects.")
        fig = axes[0].get_figure()
        own_fig = False

    plot_grid(ds, mode="logical", selection=selection, ax=axes[0])
    plot_grid(ds, mode="poloidal", selection=selection, ax=axes[1], legend=False)
    axes[0].set_title("Logical grid")
    axes[1].set_title("Poloidal grid")
    if own_fig:
        fig.tight_layout()
        fig.suptitle(title, y=1.03)
        plt.show()


def plot_grid(
    ds,
    selection=None,
    mode="logical",
    ax=None,
    xlim=(None, None),
    ylim=(None, None),
    plot_region_boundaries=True,
    legend=True,
    title="",
    linecolor="grey",
    linewidth=0.2,
):
    """
    Create a 2D polygon plot of a Hermes-3 grid

    Parameters
    ----------
    ds : dict-like
        Either a Hermes-3 results dataset or a HypnotoadGrid object. Needs to have
        metadata with region boundaries, and Rxy, Zxy and their corner coordinates
        as keys.
    selection : (radial_sel, poloidal_sel)
        Two-element tuple of int or slice specifying the row and column
        selectors, compatible with NumPy 2D indexing.
    mode : "logical" shows grid in index space. "poloidal" shows R,Z space.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, a new figure and axes will be created
    xlim : tuple, optional
        Limits for the x-axis (R). Default is (None, None) which means automatic
    ylim : tuple, optional
        Limits for the y-axis (Z). Default is (None, None) which means automatic
    linecolor : str, optional
        Color of the grid lines. Default is 'k' (black).
    linewidth : float, optional
        Width of the grid lines. Default is 0.3.
    """

    # If reading a results dataset with time, select last time slice
    if hasattr(ds, "coords"):
        if "t" in ds.sizes:
            ds = ds.isel(t=-1)

    m = ds.metadata

    if ax == None:
        fig, ax = plt.subplots()

    # Marker size for selection plot
    ms_selection = 3

    ax.set_title(title)

    cmap = mpl.colors.ListedColormap(
        [
            "white",
            "coral",
            "limegreen",
            "skyblue",
            "violet",
            "navy",
            "grey",
            "darkslategrey",
            "deeppink",
        ]
    )
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, cmap.N + 0.5, 1), cmap.N)

    # Handle different naming conventions in grid and xBOUT dataset
    if "Rxy" in ds.keys():
        Rname = "Rxy"
        Zname = "Zxy"
    elif "R" in ds.keys():
        Rname = "R"
        Zname = "Z"
    else:
        raise Exception("RZ coordinates not found in dataset")

    if mode == "poloidal":
        if "Rxy_lower_right_corners" not in ds.keys():
            raise Exception("Cell corners not present in mesh, cannot do polygon plot")

        else:
            r_nodes = [
                Rname,
                "Rxy_lower_left_corners",
                "Rxy_lower_right_corners",
                "Rxy_upper_left_corners",
                "Rxy_upper_right_corners",
            ]
            z_nodes = [
                Zname,
                "Zxy_lower_left_corners",
                "Zxy_lower_right_corners",
                "Zxy_upper_left_corners",
                "Zxy_upper_right_corners",
            ]
            cell_r = np.concatenate(
                [np.expand_dims(ds[x], axis=2) for x in r_nodes], axis=2
            )
            cell_z = np.concatenate(
                [np.expand_dims(ds[x], axis=2) for x in z_nodes], axis=2
            )

        Nx = len(cell_r)
        Ny = len(cell_r[0])
        patches = []

        idx = [np.array([1, 2, 4, 3, 1])]
        patches = []
        for i in range(Nx):
            for j in range(Ny):
                p = mpl.patches.Polygon(
                    np.concatenate((cell_r[i][j][tuple(idx)], cell_z[i][j][tuple(idx)]))
                    .reshape(2, 5)
                    .T,
                    fill=False,
                    closed=True,
                    facecolor=None,
                )
                patches.append(p)

        color_idx = np.zeros((Nx, Ny), dtype=int)

        if plot_region_boundaries:
            color_idx[:, m["jyseps1_1g"]] = 1
            color_idx[:, m["jyseps1_2g"]] = 2
            color_idx[:, m["jyseps2_1g"]] = 3
            color_idx[:, m["jyseps2_2g"]] = 4
            color_idx[:, m["ny_innerg"]] = 5
            color_idx[m["ixseps1"], :] = 6
            if "single-null" not in m["topology"]:
                color_idx[m["ixseps2"], :] = 7

            # Plot selection: color patches deeppink in RZ mode
            if selection != None:
                color_idx[selection] = 8
                ax.plot(
                    ds[Rname][selection],
                    ds[Zname][selection],
                    label="selection",
                    lw=0,
                    alpha=1,
                    ms=ms_selection / 5,
                    marker="o",
                    c=cmap(8),
                    markeredgecolor="yellow",
                    zorder=100,
                )

            ax.plot(
                ds[Rname][m["ixseps1"], :],
                ds[Zname][m["ixseps1"], :],
                label="ixseps1",
                lw=0,
                alpha=1,
                ms=2,
                marker="o",
                c=cmap(5),
            )

            ax.plot(
                ds[Rname][selection],
                ds[Zname][selection],
                label="selection",
                lw=0,
                alpha=1,
                ms=ms_selection / 5,
                marker="o",
                c=cmap(8),
                markeredgecolor="yellow",
                zorder=100,
            )

            ax.plot(
                ds[Rname][m["ixseps1"], :],
                ds[Zname][m["ixseps1"], :],
                label="ixseps1",
                lw=0,
                alpha=1,
                ms=2,
                marker="o",
                c=cmap(5),
            )

            if "single-null" not in m["topology"]:
                ax.plot(
                    ds[Rname][m["ixseps2"], :],
                    ds[Zname][m["ixseps2"], :],
                    label="ixseps2",
                    lw=0,
                    alpha=1,
                    ms=2,
                    marker="o",
                    c=cmap(6),
                )

        colors_flat = color_idx.flatten()

        polys = mpl.collections.PatchCollection(
            patches,
            alpha=1,
            norm=norm,
            cmap=cmap,
            antialiaseds=True,
            edgecolors=linecolor,
            linewidths=linewidth,
            joinstyle="bevel",
        )

        polys.set_array(colors_flat)
        ax.add_collection(polys)
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylim(cell_z.min(), cell_z.max())
        ax.set_xlim(cell_r.min(), cell_r.max())
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")

    elif mode == "logical":
        x = np.array(range(m["nxg"]))
        y = range(m["nyg"])

        X, Y = np.meshgrid(y, x)
        color = np.zeros_like(X)

        color[:, m["jyseps1_1g"]] = 1
        color[:, m["jyseps1_2g"]] = 2
        color[:, m["jyseps2_1g"]] = 3
        color[:, m["jyseps2_2g"]] = 4
        color[:, m["ny_innerg"]] = 5
        color[m["ixseps1"], :] = 6
        if "single-null" not in m["topology"]:
            color[m["ixseps2"], :] = 7

        if selection != None:
            ax.plot(
                Y[selection],
                X[selection],
                label="selection",
                lw=0,
                alpha=1,
                ms=ms_selection,
                marker="o",
                c=cmap(8),
                markeredgecolor="yellow",
                zorder=100,
            )

        ax.pcolormesh(
            Y,
            X,
            color,
            cmap=cmap,
            norm=norm,
            linewidth=0.1,
            antialiased=True,
            color="k",
        )

        ax.set_xlabel("Y index")
        ax.set_ylabel("X index")

    legend_handles = [
        mpl.lines.Line2D([0], [0], label="jyseps1_1g", color=cmap(1)),
        mpl.lines.Line2D([0], [0], label="jyseps1_2g", color=cmap(2)),
        mpl.lines.Line2D([0], [0], label="jyseps2_1g", color=cmap(3)),
        mpl.lines.Line2D([0], [0], label="jyseps2_2g", color=cmap(4)),
        mpl.lines.Line2D([0], [0], label="ny_innerg", color=cmap(5)),
        mpl.lines.Line2D([0], [0], label="ixseps1", color=cmap(6)),
        mpl.lines.Line2D([0], [0], label="ixseps2", color=cmap(7)),
        mpl.lines.Line2D(
            [0],
            [0],
            label="Selection",
            color=cmap(8),
            marker="o",
            markeredgecolor="yellow",
        ),
    ]

    if legend:
        ax.legend(handles=legend_handles, loc="best", ncols=2, fontsize="xx-small")

    ax.set_axisbelow(True)
    ax.grid(False)

    if xlim != (None, None):
        ax.set_xlim(xlim)
    if ylim != (None, None):
        ax.set_ylim(ylim)

    return ax
