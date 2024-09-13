# xHermes

xHermes is a post-processing package for Hermes-3 in 1D, 2D and 3D which provides automatic conversion to SI units and many useful plotting routines.

xHermes a wrapper around [xBOUT](https://github.com/boutproject/xBOUT) to provide [Hermes-3](https://github.com/bendudson/hermes-3) specific 
features. At the moment, xHermes is used for loading in the Hermes-3 results, performing SI unit conversion and calculating some geometry terms.
You may still want to use xBOUT, in particular if you are running 2D and 3D cases and want plotting routines for those.

There is also a collection of Mike Kryjak's personal post-processing scripts [sdtools](https://github.com/mikekryjak/sdtools/tree/main/hermes3) which have extensive additional functionality for 1D and 2D postprocessing.
These scripts are not officially supported, but feel free to take a look.
It is intended for all those additional features to be merged into xHermes over time.

Both xBOUT and xHermes use [Xarray](https://docs.xarray.dev/en/stable/) which provides a scalable and powerful framework
for dealing with large amounts of data while preserving dimensional 
consistency. If you have ever used [Pandas](https://pandas.pydata.org/), you will find Xarray very familiar, as it is effectively Pandas in more dimensions.

While Xarray is based on Numpy arrays under the hood and the data 
can be accessed in this way, Xarray's main benefit comes in its
powerful abilities to select, filter and operate on data. The 
syntax is similar to Pandas and the basics are explained in 
Xarray's excellent documentation.
If you are new to Xarray, it may be useful to start with reviewing the [selection syntax](https://docs.xarray.dev/en/stable/user-guide/indexing.html).

## Installation 

Hermes-3 is pip installable. In the terminal with your Python environment enabled, run:

    pip install xhermes

However, you may not be able to get the latest version this way. For now, it is highly recommended you do an editable install instead:

    git clone https://github.com/boutproject/xhermes
    cd xhermes
    pip install -e .

In this way, pip will install the Python package from the xHermes folder you cloned. If you use git to checkout branches or modify the contents in any way, it will be automatically work.

## Loading simulations

The data from a Hermes-3 run can be loaded with just

    import xhermes
    ds = xhermes.open(hermes_simulation_path)

where the `hermes_simulation_path` directory is assumed to contain the `BOUT.dmp.*.nc`
files and a `BOUT.settings` file. `open` returns an instance of an
`xarray.Dataset` which contains BOUT- and Hermes-specific information
in the `attrs`, so represents a general structure for storing all of
the output of a simulation, including data, input options and (soon)
grid data.

When working with 2D or 3D simulations of tokamaks, it can be useful to 
load the grid along with the simulation file and pass the "toroidal" flag to xBOUT:

    import xhermes
    ds = xhermes.open(hermes_simulation_path, gridfilepath = grid_file_path, geometry = "toroidal")

This loads the metric tensor and grid corner coordinates from the grid which
enables geometrical operations, polygonal plotting routines and additional metadata on topology.

## Working with the data

### Normalisation and geometry extraction

All variables within Hermes-3 are normalised. The normalisation factors, the units and variable names 
are provided in the Hermes-3 output for most variables. 
This information can be accessed for each variable DataArray:

    ds["Pe"].attrs

xHermes automatically applies the unnormalisation when the simulation is loaded in.

When working with 1D and 2D tokamak simulations, it can be useful to extract 
cell lengths, volumes and cross-sectional areas as well as some additional metadata from the model:

    ds = ds.hermes.extract_1D_tokamak_geometry()

or

    ds = ds.hermes.extract_2D_tokamak_geometry()

### The Dataset

The Dataset `ds` contains useful objects:
- A DataArray for each variable: `ds.data_vars` shows a list. `ds["Pe"]` returns the Pe DataArray.
- A DataArray for each coordinate: `ds.coords` shows a list. `ds["t"]` or `ds.coords["t"]` returns the t DataArray.
- A summary of dimensions in the model: `ds.dims`
- Model metadata: `ds.metadata`
- Model input settings: `ds.options`

### The DataArray

The DataArray (e.g. `ds["Pe"]`) contains the array with the variable data, its dimensionality and
attributes as well as metadata and options inherited from the parent dataset.

A visual summary of the DataArray contents and size can be obtained by simply:

    ds["Pe"]

Note that this may be slow due to the Xarray overhead. To access the underlying data as a Numpy array, one can use:

    ds["Pe"].values

or

    ds["Pe"].data

### Plotting
Xarray contains a lot of built-in plotting routines which can be used to prepare [excellent 1D and 2D plots](https://docs.xarray.dev/en/stable/user-guide/plotting.html).
Unfortunately, it is unable to deal with 2D plots in tokamak geometry.
Routines to plot those are available in xBOUT which has some excellent [example notebooks](https://github.com/boutproject/xBOUT-examples).



