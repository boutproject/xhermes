# xHermes

Extends [xBOUT](https://github.com/boutproject/xBOUT) with routines specific to
the [Hermes-3](https://github.com/bendudson/hermes-3) model.

The data from a Hermes-3 run can be loaded with just

    import xhermes
    bd = xhermes.open('./data/')

where the `data` directory is assumed to contain the `BOUT.dmp.*.nc`
files and a `BOUT.settings` file. `open` returns an instance of an
`xarray.Dataset` which contains BOUT- and Hermes-specific information
in the `attrs`, so represents a general structure for storing all of
the output of a simulation, including data, input options and (soon)
grid data.

