from xarray import register_dataset_accessor, register_dataarray_accessor
from xbout import BoutDatasetAccessor, BoutDataArrayAccessor


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
        else:
            raise ValueError("Unrecognised units_type: " + units_type)
        return self
