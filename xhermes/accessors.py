import numpy as np
from xarray import register_dataset_accessor, register_dataarray_accessor
from xbout import BoutDatasetAccessor, BoutDataArrayAccessor
from .selectors import slice_2d


@register_dataset_accessor("hermes")
class HermesDatasetAccessor(BoutDatasetAccessor):
    """
    Methods on Hermes-3 datasets
    """

    def __init__(self, ds):
        super().__init__(ds)
        
    
    def select_region(self, name):
        """
        Select a radial/poloidal region from the dataset
        
        Parameters
        ----------
        name : str
            Region name to select. Must be compatible with `slice_2d`.
        
        Returns
        -------
        xarray.Dataset
            Dataset with data selected for the specified region
        """
        selection = slice_2d(self.data, name)
        return self.data.isel(x=selection[0], theta=selection[1])

    def unnormalise(self):
        """
        In-place modify data, converting to SI units

        Normalisation values set in load.open_hermesdataset
        in attrs['conversion']
        """

        # Un-normalise data arrays
        for key, da in self.data.items():
            da.hermes.unnormalise()

        # Un-normalise coordinates
        for coord in ["dx", "dy", "dz", "t"]:
            units_type = self.data[coord].attrs.get("units_type", "unknown")
            if (units_type == "unknown") or (units_type == "SI"):
                continue
            elif units_type == "hermes":
                # Un-normalise coordinate, modifying in place
                self.data[coord] = (
                    self.data[coord] * self.data[coord].attrs["conversion"]
                )
                self.data[coord].attrs["units_type"] = "SI"
            else:
                raise ValueError("Unrecognised units_type: " + units_type)
            
    def extract_1d_tokamak_geometry(self, remove_outer = False):
        """
        Process the results to generate 1D relevant geometry data:
        - Reconstruct pos, the cell position in [m] from upstream from dy
        - Calculate cross-sectional area da and volume dv

        Notes
        ----------
        - dy is technically in flux space, but in 1D 
          this is the same as in regular space and in units of [m]
          
        Returns
        ----------
        - Dataset with the new geometry 
        """
        ds = self.data.squeeze() # Get rid of 1-length dimensions

        # Reconstruct grid position (pos, as in position) from dy
        dy = ds.coords["dy"].values
        n = len(dy)
        pos = np.zeros(n)
        pos[0] = 0.5*dy[0]

        for i in range(1,n):
            pos[i] = pos[i-1] + 0.5*dy[i-1] + 0.5*dy[i]
            
        # Set 0 to be at first cell boundary in domain
        if remove_outer is True:
            pos -= (pos[2] + pos[3]) / 2     
        else:
            pos -= (pos[1] + pos[2]) / 2   
        

        ds["pos"] = (["y"], pos)
        
        # Make pos the main coordinate instead of y
        ds = ds.swap_dims({"y":"pos"})
        ds.coords["pos"].attrs = ds.coords["y"].attrs
        
        ds["pos"].attrs.update({
            "conversion" : 1,
            "units" : "m",
            "standard_name" : "parallel position",
            "long_name" : "Parallel connection length"})
    

        # Derive and append metadata for the cross-sectional area
        # and volume. The conversions are 1 because the derivation
        # is from already-normalised parameters
        ds["da"] = ds.dx * ds.dz * ds.J / np.sqrt(ds.g_22)
        ds["da"].attrs.update({
            "conversion" : 1,
            "units" : "m2",
            "standard_name" : "cross-sectional area",
            "long_name" : "Cell parallel cross-sectional area"})
        
        ds["dv"] = ds.J * ds.dx * ds.dy * ds.dz 
        ds["dv"].attrs.update({
            "conversion" : 1,
            "units" : "m3",
            "standard_name" : "cell volume",
            "long_name" : "Cell Volume"})
        
        return ds
    
    def extract_2d_tokamak_geometry(self):
        """
        Process the results to generate 2D relevant geometry data:
        
        Calculates
        ----------
        - nxg and nyg, versions of ny and nx which always include guard cells 
          if they exist in the dataset, and do not include them if they don't
        - jyseps1_1g, jyseps1_2g, ..., versions of jyseps which account for guards in the 
          same way as nxg and nyg
        - x_idx, y_idx: arrays of radial and poloidal indices of all cells
        - dv, dr, dthe, dl: cell volume and real space cell dimensions
        - Adds target names to the metadata accounting for single or double null
        
        Notes
        ----------
        - Adds copies of jyseps1_1 that are shortened to j1_1 etc. 
          This is potentially annoying.
        - The reason it's useful to have arrays of coordinate indices
          is because Xarray is surprisingly awkward when it comes to 
          obtaining this from the coordinates.
        - This method has a hardcoded requirement for providing the grid file.
          
        Returns
        ----------
        - Dataset with the new geometry 
        """
        
        ds = self.data.squeeze()
        m = ds.metadata
        
        if "topology" not in m:
            raise Exception(
                "2D Tokamak topology missing from metadata. Please load model with the flag geometry = 'toroidal' and provide grid")
        
        # TODO: get rid of the below once xBOUT differentiates 
        # between USN and LSN
        if "single-null" in m["topology"]:
  
            if ds["Rxy"][0, m["jyseps1_1"]] < ds["Rxy"][0, m["jyseps2_2"]]:
                m["topology"] = "lower-single-null"
            
            if ds["Rxy"][0, m["jyseps1_1"]] > ds["Rxy"][0, m["jyseps2_2"]]:
                m["topology"] = "upper-single-null"
        
        # Add theta index to coords so that both X and theta can be accessed index-wise
        # It is surprisingly hard to extract the index of coordinates in Xarray...
        ds.coords["theta_idx"] = (["theta"], range(len(ds.coords["theta"])))
        
        # Extract target names. This is done here and not in load because load is not 
        # tokamak specific.
        if "single-null" in m["topology"]:
            m["targets"] = ["inner_lower", "outer_lower"]
        elif "double-null" in m["topology"]:
            m["targets"] = ["inner_lower", "outer_lower", "inner_upper", "outer_upper"]
            
        num_targets = len(m["targets"])

        # nyg, nxg: cell counts which are always with guard cells 
        # if they exist, or not if they don't
        if m["keep_xboundaries"] == 0:
            m["nxg"] = m["nx"] - m["MXG"] * 2 
            m["MXG"] = 0
        else:
            m["nxg"] = m["nx"]
            
        if m["keep_yboundaries"] == 0:
            m["nyg"] = m["ny"]    
            m["MYG"] = 0
        else:
            m["nyg"] = m["ny"] + m["MYG"] * num_targets   
        
        
        # Separatrix indices which account for guard cells in the same way as nxg, nyg
        # TODO: switch these over to xBOUT once available
        m["jyseps1_1g"] = m["jyseps1_1"] + m["MYG"]
        m["jyseps1_2g"] = m["jyseps1_2"] + m["MYG"] * (num_targets - 1)
        m["jyseps2_1g"] = m["jyseps2_1"] + m["MYG"]
        m["jyseps2_2g"] = m["jyseps2_2"] + m["MYG"] * (num_targets - 1)
        m["ny_innerg"] = m["ny_inner"] + m["MYG"] * (num_targets - 1)
            
        # Array of radial (x) indices and of poloidal (y) indices for each cell
        # This is useful because Xarray makes it awkward to extract indices in certain cases
        ds["x_idx"] = (["x", "theta"], np.array([np.array(range(m["nxg"]))] * int(m["nyg"])).transpose())
        ds["y_idx"] = (["x", "theta"], np.array([np.array(range(m["nyg"]))] * int(m["nxg"])))
        
        # Cell volume calculation
        ds["dv"] = (["x", "theta"], ds["dx"].data * ds["dy"].data * ds["dz"].data * ds["J"].data)
        ds["dv"].attrs.update({
            "conversion" : 1,
            "units" : "m3",
            "standard_name" : "cell volume",
            "long_name" : "Cell volume",
            "source" : "xHermes"})
        
        # Cell areas in real space - comes from Jacobian
        # Note: can be calculated from flux space or real space coordinates:
        # dV = (hthe/Bpol) * (R*Bpol*dr) * dy*2pi = hthe * dy * dr * 2pi * R

        # Radial cell length
        ds["dr"] = (["x", "theta"], ds.dx.data / (ds.R.data * ds.Bpxy.data))
        ds["dr"].attrs.update({
            "conversion" : 1,
            "units" : "m",
            "standard_name" : "radial length",
            "long_name" : "Length of cell in the radial direction",
            "source" : "xHermes"})
        
        ds["hthe"] = (["x", "theta"], ds["J"].data * ds["Bpxy"].data) 
        ds["hthe"].attrs.update({
            "conversion" : 1,
            "units" : "m/radian",
            "standard_name" : "h_theta: poloidal arc length per radian",
            "long_name" : "h_theta: poloidal arc length per radian",
            "source" : "xHermes"})
        

        # Toroidal cell length
        ds["dtor"] = (
            ["x", "theta"],
            ds["dz"].data * np.sqrt(ds["g_33"].data),
        )  
        ds["dtor"].attrs.update(
            {
                "conversion": 1,
                "units": "m",
                "standard_name": "Toroidal length",
                "long_name": "Toroidal length",
                "source": "xHermes",
            }
        )

        # Poloidal cell length
        ds["dpol"] = (
            ["x", "theta"],
            ds["dy"].data * ds["hthe"].data,
        )  # Poloidal length
        ds["dpol"].attrs.update(
            {
                "conversion": 1,
                "units": "m",
                "standard_name": "Poloidal length",
                "long_name": "Poloidal length",
                "source": "xHermes",
            }
        )

        return ds

    def calculate_boundary_flows(self):
        """
        Calculate particle sources at SOL and PFR boundaries

        For each species in the dataset, calculates the particle source
        at the SOL and PFR boundaries by dividing the flow diagnostics
        by cell volume. These are then added back to the dataset.
        This allows boundary flows to be treated like regular sources
        which helps make post-processing more straightforward.

        TODO: Add energy flows
        TODO: Add flows in y direction

        Returns
        -------
        xarray.Dataset
            Dataset with added boundary flow source terms
        """

        ds = self.data

        sol = ds.hermes.select_region("sol_boundary")
        pfr = ds.hermes.select_region("pfr_boundary")
        sol_guard = ds.hermes.select_region("sol_boundary_guard")

        for name in ds.metadata["species"]:

            # Account for inconsistency in diagnostic names
            if name in ds.metadata["neutral_species"]:
                flow_diagnostic_name = f"pf{name}_adv_perp_xlow"
            else:
                flow_diagnostic_name = f"pf{name}_tot_xlow"

            if flow_diagnostic_name in ds.data_vars:

                # xlow means SOL boundary flow is read at guard cell
                ds[f"S{name}_sol_boundary"] = (
                    sol_guard[flow_diagnostic_name] / sol["dv"]
                )
                ds[f"S{name}_sol_boundary"].attrs.update(
                    {
                        "short_name": "Particle source",
                        "long_name": f"Particle source of {name} at SOL boundary",
                        "units": "m^-3 s^-1",
                    }
                )

                # xlow means PFR boundary flow is read at final domain cell
                ds[f"S{name}_pfr_boundary"] = pfr[flow_diagnostic_name] / pfr["dv"]
                ds[f"S{name}_pfr_boundary"].attrs.update(
                    {
                        "short_name": "Particle source",
                        "long_name": f"Particle source of {name} at PFR boundary",
                        "units": "m^-3 s^-1",
                    }
                )

        return ds

@register_dataarray_accessor("hermes")
class HermesDataArrayAccessor(BoutDataArrayAccessor):
    """
    Methods on Hermes-3 arrays
    """

    def __init__(self, da):
        super().__init__(da)
        
    def clear_guards(self):
        """
        Set guard cell values to np.nan
        """
        
        # Clear radial guards
        xguards = slice_2d(self.data, "xguards")
        ds = self.data.copy()
        ds[{"x": xguards[0], "theta": xguards[1]}] = np.nan
        
        # Clear target guards if they exist
        if self.data.metadata["MYG"] > 0:
            yguards = slice_2d(self.data, f"yguards")
            ds[{"x": yguards[0], "theta": yguards[1]}] = np.nan
            
        return ds
        
    def select_region(self, name):
        """
        Select a radial/poloidal region from the DataArray
        
        Parameters
        ----------
        name : str
            Region name to select. Must be compatible with `slice_2d`.
        
        Returns
        -------
        xarray.DataArray
            DataArray with data selected for the specified region
        """
        selection = slice_2d(self.data, name)
        return self.data.isel(x=selection[0], theta=selection[1])

    def unnormalise(self):
        """
        In-place modify data, converting to SI units

        Normalisation values set in load.open_hermesdataset
        in attrs['conversion']
        """
        units_type = self.data.attrs.get("units_type", "unknown")

        if units_type == "SI":
            # already un-normalised
            return
        elif ("units" in self.data.attrs) and ("conversion" in self.data.attrs):
            # Normalise using values
            self.data *= self.data.attrs["conversion"]
            self.data.attrs["units_type"] = "SI"
        return self
