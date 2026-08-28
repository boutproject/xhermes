import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import xbout

from .selectors import selector_2d, get_sepx_coords


def plot_selection(
    ds,
    radial_region=None,
    poloidal_region=None,
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
        if (
            not isinstance(custom_selection, (tuple, list))
            or len(custom_selection) != 2
        ):
            raise ValueError(
                "custom_selection must be a 2-item tuple/list of (radial_sel, poloidal_sel)."
            )

    if custom_selection is not None:
        selection = custom_selection
    else:
        selection = selector_2d(ds, radial_region, poloidal_region)

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 4), dpi=dpi)
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
    plot_regions=False,
    legend=True,
    title="",
    linecolor="black",
    linewidth=0.05,
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

    if ax is None:
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

    if plot_regions and plot_region_boundaries:
        raise ValueError(
            "Cannot plot both regions and region boundaries at the same time, please choose one or the other."
        )

    def apply_color(color_idx):
        if "single" in ds.metadata["topology"]:
            color_idx[selector_2d(ds, "sol", "sol")] = 1
            color_idx[selector_2d(ds, "sol", "inner_divertor")] = 2
            color_idx[selector_2d(ds, "sol", "outer_divertor")] = 3
            color_idx[selector_2d(ds, "core", "pfr")] = 4
            color_idx[selector_2d(ds, "core", "core")] = 5

        elif "double" in ds.metadata["topology"]:
            color_idx[selector_2d(ds, "sol", "sol")] = 1
            color_idx[selector_2d(ds, "sol", "inner_lower_divertor")] = 2
            color_idx[selector_2d(ds, "sol", "inner_upper_divertor")] = 3
            color_idx[selector_2d(ds, "sol", "outer_upper_divertor")] = 4
            color_idx[selector_2d(ds, "sol", "outer_lower_divertor")] = 5
            color_idx[selector_2d(ds, "core", "pfr")] = 6
            color_idx[selector_2d(ds, "core", "core")] = 7

        else:
            raise ValueError(f"Unknown topology: {ds.metadata['topology']}")

        return color_idx

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

        if plot_regions:
            color_idx = apply_color(color_idx)

        if plot_region_boundaries:
            color_idx[:, m["jyseps1_1g"]] = 1
            color_idx[:, m["jyseps1_2g"]] = 2
            color_idx[:, m["jyseps2_1g"]] = 3
            color_idx[:, m["jyseps2_2g"]] = 4
            color_idx[:, m["ny_innerg"]] = 5
            color_idx[m["ixseps1g"], :] = 6
            if "single-null" not in m["topology"]:
                color_idx[m["ixseps2g"], :] = 7

            # Plot selection: color patches deeppink in RZ mode
            if selection is not None:
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

            # Plot separatrix
            ax.plot(
                ds[Rname][m["ixseps1g"], :],
                ds[Zname][m["ixseps1g"], :],
                label="ixseps1g",
                lw=0,
                alpha=1,
                ms=2,
                marker=None,
                c=cmap(5),
            )

            if "single-null" not in m["topology"]:
                ax.plot(
                    ds[Rname][m["ixseps2g"], :],
                    ds[Zname][m["ixseps2g"], :],
                    label="ixseps2g",
                    lw=0,
                    alpha=1,
                    ms=2,
                    marker=None,
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

        if plot_regions:
            color_idx = apply_color(color)

        if plot_region_boundaries:
            color[:, m["jyseps1_1g"]] = 1
            color[:, m["jyseps1_2g"]] = 2
            color[:, m["jyseps2_1g"]] = 3
            color[:, m["jyseps2_2g"]] = 4
            color[:, m["ny_innerg"]] = 5
            color[m["ixseps1g"], :] = 6
            if "single-null" not in m["topology"]:
                color[m["ixseps2g"], :] = 7

        if selection is not None:
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
            linewidth=linewidth,
            antialiased=True,
            color="k",
        )

        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")

    if plot_region_boundaries:
        legend_handles = [
            mpl.lines.Line2D([0], [0], label="jyseps1_1g", color=cmap(1)),
            mpl.lines.Line2D([0], [0], label="jyseps1_2g", color=cmap(2)),
            mpl.lines.Line2D([0], [0], label="jyseps2_1g", color=cmap(3)),
            mpl.lines.Line2D([0], [0], label="jyseps2_2g", color=cmap(4)),
            mpl.lines.Line2D([0], [0], label="ny_innerg", color=cmap(5)),
            mpl.lines.Line2D([0], [0], label="ixseps1g", color=cmap(6)),
            mpl.lines.Line2D([0], [0], label="ixseps2g", color=cmap(7)),
            mpl.lines.Line2D(
                [0],
                [0],
                label="Selection",
                color=cmap(8),
                marker="o",
                markeredgecolor="yellow",
            ),
        ]

    else:
        if "single" in ds.metadata["topology"]:
            legend_handles = [
                mpl.lines.Line2D([0], [0], label="SOL", color=cmap(1)),
                mpl.lines.Line2D([0], [0], label="Inner divertor", color=cmap(2)),
                mpl.lines.Line2D([0], [0], label="Outer divertor", color=cmap(3)),
                mpl.lines.Line2D([0], [0], label="PFR", color=cmap(4)),
                mpl.lines.Line2D([0], [0], label="Core", color=cmap(5)),
                # mpl.lines.Line2D([0], [0], label="ixseps1g", color=cmap(6)),
                # mpl.lines.Line2D([0], [0], label="ixseps2g", color=cmap(7)),
                mpl.lines.Line2D(
                    [0],
                    [0],
                    label="Selection",
                    color=cmap(8),
                    marker="o",
                    markeredgecolor="yellow",
                ),
            ]

        elif "double" in ds.metadata["topology"]:
            legend_handles = [
                mpl.lines.Line2D([0], [0], label="SOL", color=cmap(1)),
                mpl.lines.Line2D([0], [0], label="Inner divertor", color=cmap(2)),
                mpl.lines.Line2D([0], [0], label="Outer divertor", color=cmap(3)),
                mpl.lines.Line2D([0], [0], label="PFR", color=cmap(4)),
                mpl.lines.Line2D([0], [0], label="Core", color=cmap(5)),
                # mpl.lines.Line2D([0], [0], label="ixseps1g", color=cmap(6)),
                # mpl.lines.Line2D([0], [0], label="ixseps2g", color=cmap(7)),
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


def animate2d(da, savepath=None, fps=10, separatrix=True, cbar: bool = True, **kwargs):

    if "ax" not in kwargs.keys():
        fig, ax = plt.subplots(figsize=(4, 6), dpi=120)
        kwargs["ax"] = ax

    # TODO: Add a better separatrix plotting routine that tracaes along cell corners, not cell centres
    if separatrix:
        # xbout.plotting.utils.plot_separatrices(da, ax)
        sepx_R, sepx_Z = get_sepx_coords(da)
        ax.plot(sepx_R, sepx_Z,color="gray",linestyle="--",linewidth=1.0)

    if set(da.dims) == set(["t", "x", "theta"]):
        slider = plot2d_polygon_with_time_slider(
            da, savepath=savepath, fps=fps, cbar=cbar, **kwargs
        )
        return slider
    elif set(da.dims) == set(["x", "theta"]):
        # Extract some grid information
        rm = np.stack(
            [
                da.Rxy_lower_left_corners,
                da.Rxy_upper_left_corners,
                da.Rxy_upper_right_corners,
                da.Rxy_lower_right_corners,
            ]
        ).transpose(1, 2, 0)
        zm = np.stack(
            [
                da.Zxy_lower_left_corners,
                da.Zxy_upper_left_corners,
                da.Zxy_upper_right_corners,
                da.Zxy_lower_right_corners,
            ]
        ).transpose(1, 2, 0)
        nx = rm.shape[0]
        ny = rm.shape[1]
        _ = plot2d_polygon(da.values, rm, zm, nx, ny, **kwargs)
        return None
    else:
        raise ValueError(
            "Input da must have either dimensions (t, x, theta) or (x, theta)"
        )


def plot2d_polygon(
    vals,
    rm,
    zm,
    nx,
    ny,
    ax=None,
    vmin: float = None,
    vmax: float = None,
    cmap="magma",
    logscale: bool = False,
    lw: float = 0.0,
    linthresh: float = 1.0,
):
    """2D polygon plot in poloidal geometry. This is an alternative to xbout.polygon. Input da is assumed to contain only two dimensions: R and Z"""

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6), dpi=120)
    else:
        fig = ax.get_figure()
    ax.set_aspect("equal")

    # TODO: Include option to plot separatrix, targets, etc as in plot_grid()

    if vmin is None:
        vmin = np.min(vals)
    if vmax is None:
        vmax = np.max(vals)

    patches = []
    for iy in np.arange(0, ny):
        for ix in np.arange(0, nx):
            rcol = rm[ix, iy, :]
            zcol = zm[ix, iy, :]
            polygon = Polygon(np.column_stack((rcol, zcol)))
            patches.append(polygon)

    patch_vals = vals.transpose().flatten()

    if logscale:
        if vmin < 0:
            norm = mpl.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=linthresh)
        else:
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    if lw != 0.0:
        edgecolor = "black"
        antialiased = True 
    else:
        edgecolor = "none"
        antialiased = False
    p = PatchCollection(
        patches,
        norm=norm,
        cmap=cmap,
        edgecolor=edgecolor,
        linewidth=lw,
        antialiased=antialiased,
    )

    p.set_array(np.array(patch_vals))

    ax.add_collection(p)
    ax.autoscale_view()

    return p


def plot2d_polygon_with_time_slider(da, savepath=None, fps=10, cbar=True, **kwargs):
    """2D polygon plot in poloidal geometry with a time slider. Input da is assumed to contain three dimensions: t, R and Z"""
    # Extract some grid information
    rm = np.stack(
        [
            da.Rxy_lower_left_corners,
            da.Rxy_upper_left_corners,
            da.Rxy_upper_right_corners,
            da.Rxy_lower_right_corners,
        ]
    ).transpose(1, 2, 0)
    zm = np.stack(
        [
            da.Zxy_lower_left_corners,
            da.Zxy_upper_left_corners,
            da.Zxy_upper_right_corners,
            da.Zxy_lower_right_corners,
        ]
    ).transpose(1, 2, 0)
    nx = rm.shape[0]
    ny = rm.shape[1]

    all_vals = da.values

    ax = kwargs["ax"]
    fig = ax.get_figure()

    p = plot2d_polygon(
        all_vals[0, :, :],
        rm,
        zm,
        nx,
        ny,
        vmin=np.min(all_vals),
        vmax=np.max(all_vals),
        **kwargs,
    )
    p = [p]

    if cbar:
        try:
            cbar_label = da.name + " [" + da.attrs.get("units", "") + "]"
        except:
            cbar_label = ""
        fig.colorbar(p[0], ax=ax, label=cbar_label)


    def update_patch_values(time):
        timestep = np.argmin(np.abs(da.t.values - time / 1e6))
        p[0].remove()
        p[0].set_array(all_vals[timestep, :, :].transpose().flatten())
        ax.add_collection(p[0])
        if savepath is not None:
            ax.set_title(r"timestep = {:.1f} $\mu$s".format(time))

        return p

    if savepath is not None:
        anim = animation.FuncAnimation(
            fig,
            update_patch_values,
            frames=da.t.values * 1e6,
        )
        anim.save(savepath, fps=fps)
        plt.show()
        return anim

    else:
        ax_time_slider = fig.add_axes([0.2, 0.05, 0.55, 0.03])
        time_slider = Slider(
            ax=ax_time_slider,
            label=r"Time [$\mu$s]",
            valmin=1e6 * da.t.min().values,
            valmax=1e6 * da.t.max().values,
            valinit=1e6 * da.t.min().values,
            valstep=1e6 * da.t.values,
        )
        time_slider.on_changed(update_patch_values)

        return time_slider