from xarray import register_dataset_accessor, register_dataarray_accessor
from xbout import BoutDatasetAccessor, BoutDataArrayAccessor
import numpy as np


@register_dataset_accessor("hermes")
class HermesDatasetAccessor(BoutDatasetAccessor):
    """
    Methods on Hermes-3 datasets
    """

    def __init__(self, ds):
        super().__init__(ds)

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
        for coord in ["t"]:
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
            
    def extract_1d_tokamak_geometry(self):
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
        pos[0] = -0.5*dy[1]
        pos[1] = 0.5*dy[1]

        for i in range(2,n):
            pos[i] = pos[i-1] + 0.5*dy[i-1] + 0.5*dy[i]

        # Guard replace to get position at boundaries
        pos[-2] = (pos[-3] + pos[-2])/2
        pos[1] = (pos[1] + pos[2])/2 

        # Set 0 to be at first cell boundary in domain
        pos = pos - pos[1]
        ds["pos"] = (["y"], pos)
        
        # Make pos the main coordinate instead of y
        ds = ds.swap_dims({"y":"pos"})
        ds.coords["pos"].attrs = ds.coords["y"].attrs

        # Derive and append metadata for the cross-sectional area
        # and volume. The conversions are 1 because the derivation
        # is from already-normalised parameters
        ds["da"] = ds.J / np.sqrt(ds.g_22)
        ds["da"].attrs.update({
            "conversion" : 1,
            "units" : "m2",
            "standard_name" : "cross-sectional area",
            "long_name" : "Cell parallel cross-sectional area"})
        
        ds["dv"] = ds.J * ds.dy
        ds["dv"].attrs.update({
            "conversion" : 1,
            "units" : "m3",
            "standard_name" : "cell volume",
            "long_name" : "Cell Volume"})
        
        return ds


@register_dataarray_accessor("hermes")
class HermesDataArrayAccessor(BoutDataArrayAccessor):
    """
    Methods on Hermes-3 arrays
    """

    def __init__(self, da):
        super().__init__(da)

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
