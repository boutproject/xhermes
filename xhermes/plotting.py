import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_rz_grid(ds, ax = None, 
                     xlim = (None,None), ylim = (None,None),
                     plot_region_boundaries = True,
                     title = "", linecolor = "grey", linewidth = 0.2):
    
    """
    Create a 2D polygon plot of a Hermes-3 grid
    
    Parameters
    ----------
    ds : dict-like
        Either a Hermes-3 results dataset or a HypnotoadGrid object. Needs to have
        metadata with region boundaries, and Rxy, Zxy and their corner coordinates
        as keys.
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
    
    m = ds.metadata
    
    if ax == None:
        fig, ax = plt.subplots()

    ax.set_title(title)
    
    
    if "Rxy_lower_right_corners" in ds.keys():
        r_nodes = [
            "Rxy",
            "Rxy_corners",
            "Rxy_lower_right_corners",
            "Rxy_upper_left_corners",
            "Rxy_upper_right_corners",
        ]
        z_nodes = [
            "Zxy",
            "Zxy_corners",
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
    else:
        raise Exception("Cell corners not present in mesh, cannot do polygon plot")

    Nx = len(cell_r)
    Ny = len(cell_r[0])
    patches = []

    # https://matplotlib.org/2.0.2/examples/api/patch_collection.html

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
            
    # cmap = mpl.colors.ListedColormap(["white"])
    cmap = mpl.colors.ListedColormap(["white", "red", "green", "blue", "purple", "red", "deeppink"])
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, cmap.N + 0.5, 1), cmap.N)
    
    color_idx = np.zeros((Nx, Ny), dtype=int)
    
    if plot_region_boundaries:
        color_idx[:, m["j1_1g"]] = 1
        color_idx[:, m["j1_2g"]] = 2
        color_idx[:, m["j2_1g"]] = 3
        color_idx[:, m["j2_2g"]] = 4
        color_idx[m["ixseps1"], :] = 5
        color_idx[m["ixseps2"], :] = 6
        
        ax.plot(ds.Rxy[m["ixseps1"],:], ds.Zxy[m["ixseps1"],:], 
                label = "ixseps1", lw = 0, alpha = 1, ms = 2, marker = "o", c = cmap(5))
        ax.plot(ds.Rxy[m["ixseps2"],:], ds.Zxy[m["ixseps2"],:], 
                label = "ixseps2", lw = 0, alpha = 1, ms = 2, marker = "o", c = cmap(6))


    colors_flat = color_idx.flatten()
    
    polys = mpl.collections.PatchCollection(
        patches,
        alpha=1,
        norm=norm,
        cmap=cmap,
        # fill = False,
        antialiaseds=True,
        edgecolors=linecolor,
        linewidths=linewidth,
        joinstyle="bevel",
    )

    polys.set_array(colors_flat)
    ax.add_collection(polys)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_ylim(cell_z.min(), cell_z.max())
    ax.set_xlim(cell_r.min(), cell_r.max())
    
    ax.set_axisbelow(True)
    ax.grid(False)

    if xlim != (None,None):
        ax.set_xlim(xlim)
    if ylim != (None,None):
        ax.set_ylim(ylim)