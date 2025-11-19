import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_rz_grid(mesh, ax = None, 
                     xlim = (None,None), ylim = (None,None),
                     linecolor = "k", linewidth = 0.3):
    
    """
    Create a 2D polygon plot of a Hermes-3 grid
    
    Parameters
    ----------
    mesh : dict-like
        A dictionary-like object containing the mesh data. Must include keys for
        the RZ coordinates of cell centres and corners.
        Should work on both a loaded grid file or a Hermes-3 dataset.
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
    
    
    if ax == None:
        fig, ax = plt.subplots()

    ax.set_title("R, Z space")
    
    
    if "Rxy_lower_right_corners" in mesh.keys():
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
            [np.expand_dims(mesh[x], axis=2) for x in r_nodes], axis=2
        )
        cell_z = np.concatenate(
            [np.expand_dims(mesh[x], axis=2) for x in z_nodes], axis=2
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
            
    cmap = mpl.colors.ListedColormap(["white"])
    colors =np.zeros_like(cell_r).flatten()
    polys = mpl.collections.PatchCollection(
        patches,
        alpha=0.5,
        # norm=norm,
        cmap=cmap,
        # fill = False,
        antialiaseds=True,
        edgecolors=linecolor,
        linewidths=linewidth,
        joinstyle="bevel",
    )

    polys.set_array(colors)
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