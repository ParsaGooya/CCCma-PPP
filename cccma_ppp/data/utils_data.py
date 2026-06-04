import numpy as np
import xarray as xr
import dataclasses
from src.cccma_ppp.preprocessing.preprocessing import PreprocessingPipeline
import os
from pathlib import Path
import glob
from typing import final, ClassVar


@dataclasses.dataclass
class infoclass:
    """
    Container for dataset metadata including sizes, coordinates, and temporal range.
    """

    sizes: dict | None
    start_year: xr.DataArray | np.ndarray | str | int | None
    final_year: xr.DataArray | np.ndarray | str | int | None
    coords: dict
    spatial_mask: xr.Dataset = None


@dataclasses.dataclass
class ModelDataConfig:
    """
    Configuration for model dataset loading and preprocessing.
    """

    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
    file_type: str = "*.nc"
    rename_dict: dict = None
    TYPE: ClassVar[str] = "model"

    def __post_init__(self):
        """
        Validate configuration and initialize dataset metadata.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If dataset structure or configuration is invalid.
        """

        self._check_ensemble = False
        if self.ensemble_list is not None:
            self._check_ensemble = True

        _check_data(self)
        self.info = _get_ds_info(self)
        self.year_range = np.arange(
            self.info.start_year,
            self.info.final_year + self.info.sizes["lead_time"] // 12,
        )

    @final
    @classmethod
    def _allowed_dims(cls):
        """
        Return allowed dataset dimensions.

        Returns
        -------
        list of str
            Valid dimension names.
        """

        return ["year", "lead_time", "ensembles", "lat", "lon"]

    @final
    @classmethod
    def _required_dims(cls):
        """
        Return required dataset dimensions.

        Returns
        -------
        list of str
            Required dimension names.
        """

        return ["lead_time", "ensembles", "lat", "lon"]


@dataclasses.dataclass
class ObsDataConfig:
    """
    Configuration for observation dataset loading and preprocessing.
    """

    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
    file_type: str = "*.nc"
    rename_dict: dict = None
    TYPE: ClassVar[str] = "observation"

    def __post_init__(self):
        """
        Validate configuration and initialize dataset metadata.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If dataset structure or configuration is invalid.
        """

        self._check_ensemble = False
        if self.ensemble_list is not None:
            self._check_ensemble = True

        _check_data(self)
        self.info = _get_ds_info(self)
        self.year_range = np.arange(self.info.start_year, self.info.final_year + 1)

    @final
    @classmethod
    def _allowed_dims(cls):
        """
        Return allowed dataset dimensions.

        Returns
        -------
        list of str
            Valid dimension names.
        """

        return ["year", "month", "ensembles", "lat", "lon"]

    @final
    @classmethod
    def _required_dims(cls):
        """
        Return required dataset dimensions.

        Returns
        -------
        list of str
            Required dimension names.
        """

        return ["month", "lat", "lon"]


@dataclasses.dataclass
class ConditionDataConfig:
    """
    Configuration for conditioning dataset loading and preprocessing.
    """

    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
    file_type: str = "*.nc"
    rename_dict: dict = None
    TYPE: ClassVar[str] = "model"

    def __post_init__(self):
        """
        Validate configuration and initialize dataset metadata.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If dataset structure or configuration is invalid.
        """

        self._check_ensemble = False
        if self.ensemble_list is not None:
            self._check_ensemble = True

        _check_data(self)
        self.info = _get_ds_info(self)
        if self.info.start_year is not None and self.info.final_year is not None:
            self.year_range = np.arange(
                self.info.start_year,
                self.info.final_year + self.info.sizes["lead_time"] // 12,
            )

    @final
    @classmethod
    def _allowed_dims(cls):
        """
        Return allowed dataset dimensions.

        Returns
        -------
        list of str
            Valid dimension names.
        """

        return ["year", "lead_time", "ensembles", "lat", "lon"]

    @final
    @classmethod
    def _required_dims(cls):
        """
        Return required dataset dimensions.

        Returns
        -------
        list of str
            Required dimension names.
        """

        return ["lat", "lon"]


def _check_data(dataconfig: ModelDataConfig | ObsDataConfig) -> None:
    """
    Validate dataset structure and required variables.

    Parameters
    ----------
    dataconfig : ModelDataConfig or ObsDataConfig
        Dataset configuration.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data files are missing.
    ValueError
        If required variables are missing.
    AssertionError
        If dataset dimensions are invalid.
    """

    if not Path(dataconfig.paths).exists():
        raise FileNotFoundError(
            "The following file does not exist:\n" + "\n".join(dataconfig.paths)
        )

    list_paths = glob.glob(str(Path(dataconfig.paths).joinpath(dataconfig.file_type)))

    if len(list_paths) == 0:
        raise FileNotFoundError(
            f"The following file does is empty for {dataconfig.file_type} file type :\n"
            + "\n".join(dataconfig.paths)
        )

    for p in list_paths:
        with xr.open_dataset(Path(p)) as ds:
            if dataconfig.rename_dict is not None:
                ds = ds.rename(dataconfig.rename_dict)

            for dim in dataconfig._required_dims():
                assert dim in ds.dims, (
                    f"{dataconfig.TYPE} data must have {dataconfig._required_dims()} dimensions. Current dims : {ds.dims}"
                )

            if dataconfig._check_ensemble:
                assert "ensembles" in ds.dims, (
                    "Cannot select ensemble_list as ensembles dim does not exist"
                )

            for dim in ds.dims:
                assert dim in dataconfig._allowed_dims(), (
                    f'"{dim}" not a valid dimension for {dataconfig.TYPE} data: {dataconfig._allowed_dims()}'
                )
                assert dim in ds.coords, (
                    f'"coordinates for {dim} dimension does not exist. Available coords: {dict(ds.coords).keys()}'
                )

            missing = [name for name in dataconfig.names if name not in ds.data_vars]
            if missing:
                raise ValueError(f"{p} is missing variables: {missing}")
    dataconfig.list_paths = list_paths


def _get_ds_info(dataconfig: ModelDataConfig | ObsDataConfig) -> infoclass:
    """
    Extract metadata information from dataset.

    Parameters
    ----------
    dataconfig : ModelDataConfig or ObsDataConfig
        Dataset configuration.

    Returns
    -------
    infoclass
        Metadata container.
    """

    if getattr(dataconfig, "list_paths", None) is None:
        list_paths = glob.glob(
            str(Path(dataconfig.paths).joinpath(dataconfig.file_type))
        )
    else:
        list_paths = dataconfig.list_paths

    ds = _load_xarray_data(
        list_paths,
        names=dataconfig.names,
        concat_dim=dataconfig.concat_dim,
        rename_dict=dataconfig.rename_dict,
    )

    if dataconfig.ensemble_list is not None:
        ds = ds.sel(ensembles=dataconfig.ensemble_list)

    if "year" in ds.dims:
        start_year, final_year = ds.year.min().values, ds.year.max().values
    else:
        start_year = final_year = None

    sizes = {
        dim: dict(ds.sizes).get(dim)
        for dim in dict(ds.sizes).keys()
        if dim not in ["ensembles", "lat", "lon"]
    }
    if not sizes:
        sizes = None

    coords = {
        dim: dict(ds.coords).get(dim, None) for dim in ["ensembles", "lat", "lon"]
    }

    ds.close()
    del ds

    return infoclass(
        start_year=start_year, final_year=final_year, sizes=sizes, coords=coords
    )


@dataclasses.dataclass
class WeightsConfig:
    """
    Configuration for computing spatial and variable weights.
    """

    spatial_method: str = "uniform"
    variable_weights: dict[str, float] | None = None
    load_dir: str | None = None

    def __post_init__(self):
        """
        Validate weight configuration parameters.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If spatial method is invalid.
        """

        if self.load_dir is None:
            assert self.spatial_method.lower() in ["cosine_lat", "uniform"]

    def build_weights(
        self,
        target_coords=None,
        oceannanremover=None,
        save=True,
        save_path=None,
        save_name=None,
    ):
        """
        Construct spatial and variable weights.

        Parameters
        ----------
        target_coords : dict, optional
            Target coordinate mapping.
        oceannanremover : Oceannanremove, optional
            Ocean mask processor.
        save : bool, optional
            Whether to save weights.
        save_path : Path or str, optional
            Directory for saving.
        save_name : str, optional
            Filename.

        Returns
        -------
        xr.DataArray
            Computed weights.

        Raises
        ------
        AssertionError
            If loaded weights are inconsistent with expected coordinates.
        FileNotFoundError
            If load path is invalid.
        """

        if self.load_dir is not None:
            weights = xr.open_dataset(self.load_dir)
            if isinstance(weights, xr.Dataset):
                weights = _unwrape_data_variables(weights)

            if "ref" in weights.dims and {"lat", "lon"}.issubset(weights.coords):
                weights = weights.set_index(ref=["lat", "lon"])

            if oceannanremover is not None:
                assert weights.coords["ref"].equals(oceannanremover.final_locations)
            else:
                assert weights.coords["lat"].equals(target_coords["lat"])
                assert weights.coords["lon"].equals(target_coords["lon"])

        else:
            weights = np.cos(
                np.ones_like(target_coords["lon"])
                * (np.deg2rad(target_coords["lat"].to_numpy()))[..., None]
            )
            weights = xr.DataArray(weights, coords=target_coords, name="weights")

            if self.spatial_method == "uniform":
                weights = xr.ones_like(weights)

            if self.variable_weights is not None:
                weights = (
                    xr.DataArray(
                        list(self.variable_weights.values()),
                        coords={"channels": list(self.variable_weights.keys())},
                    )
                    * weights
                )

            if oceannanremover is not None:
                weights = oceannanremover.transform(weights)

            if save:
                save_path = (
                    Path(save_path)
                    if save_path is not None
                    else Path(os.environ["GLOBAL_EXP_DIR"])
                )
                save_name = save_name or "spatial_weights.nc"

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)

                weights.reset_index("ref").to_netcdf(save_path / save_name)

        return weights


def _unwrape_data_variables(dataset: xr.Dataset):
    """
    Convert dataset variables into a stacked channel dimension.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.DataArray
        Concatenated data array.
    """

    return xr.concat(
        [dataset[v].expand_dims("channels", axis=0) for v in list(dataset.data_vars)],
        dim="channels",
    )


def _load_xarray_data(
    paths,
    selection=None,
    names=None,
    ensemble_mean=False,
    preprocessor=None,
    concat_dim="year",
    rename_dict=None,
):
    """
    Load and preprocess xarray datasets from file paths.

    Parameters
    ----------
    paths : list of str
        File paths.
    selection : dict, optional
        Subselection of dataset.
    names : list, optional
        Variables to extract.
    ensemble_mean : bool, optional
        Whether to average ensembles.
    preprocessor : PreprocessingPipeline, optional
        Preprocessing pipeline.
    concat_dim : str, optional
        Concatenation dimension.
    rename_dict : dict, optional
        Variable renaming mapping.

    Returns
    -------
    xr.Dataset
        Loaded dataset.
    """

    ds = xr.open_mfdataset(paths, combine="nested", concat_dim=concat_dim)

    if rename_dict is not None:
        ds = ds.rename(rename_dict)

    ds = ds.sel(selection) if selection is not None else ds

    if "ensembles" in ds.coords:
        ds = ds.mean("ensembles") if ensemble_mean else ds

    if names is not None:
        ds = ds[names]
    if preprocessor is not None:
        ds = preprocessor.transform(ds)

    return ds


def _create_train_mask(years, lead_times, exclude_idx=0):
    """
    Create a boolean mask for training data filtering.

    Parameters
    ----------
    years : array-like
        Year indices.
    lead_times : array-like or int
        Lead time indices.
    exclude_idx : int, optional
        Offset index for exclusion.

    Returns
    -------
    xr.DataArray
        Boolean mask array.
    """

    if isinstance(lead_times, int):
        lead_times = np.arange(1, lead_times + 1)

    mask = np.full((len(years), len(lead_times)), False, dtype=bool)
    x = np.arange(0, 12 * mask.shape[0], 12)
    y = np.arange(1, mask.shape[1] + 1)
    idx_array = x[..., None] + y
    # mask[idx_array >= idx_array[-1, exclude_idx + 12]] = True
    mask[idx_array > idx_array[-1, exclude_idx + 11]] = True

    return xr.DataArray(
        mask,
        dims=("year", "lead_time"),
        coords={"year": years, "lead_time": lead_times},
        name="mask",
    )
