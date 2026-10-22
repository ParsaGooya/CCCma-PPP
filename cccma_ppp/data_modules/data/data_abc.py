import abc
from typing import final, ClassVar, Any
from collections.abc import Mapping
from pathlib import Path
import gc
import glob
import xarray as xr
import numpy as np
import pandas as pd
import dataclasses

from cccma_ppp.preprocessing.preprocessing import PreprocessingPipeline
from cccma_ppp.configs import (
    required_sample_dimensions,
    realization_dim,
    supported_NN_dimensions_sorted,
    lead_time_resolution,
)

from cccma_ppp.data_modules.utils import (
    TEMPORAL_ORDER,
    _load_xarray_data,
    _create_train_mask,
    _validate_time_sequence,
    infer_time_resolution,
    get_time_representation,
    TimeTypes,
    TimeFrequency,
    TEMPORAL_ORDER,
)
from cccma_ppp.generic.runtime import RuntimeContext


init_time_dim, lead_time_dim = required_sample_dimensions


@dataclasses.dataclass
class infoclass:
    """
    Container for xarray dataset metadata. Can be used
    to configure data loading, align datasets, and manage time coordinates
    throughout the pipeline.

    Parameters
    ----------
    sizes : dict | None
        Mapping of dimension names to their corresponding sizes. If None,
        dimension sizes are not specified.
    start_time : xr.DataArray | np.ndarray | str | int | None
        Earliest time covered by the dataset. May be provided as an xarray
        DataArray, NumPy array, date string, or integer representing a year.
        If None, the starting time is unspecified.
    final_time : xr.DataArray | np.ndarray | str | int | None
        Latest time covered by the dataset. Accepts the same formats as
        `start_time`. If None, the ending time is unspecified.
        For model datasets, this is typically the last prediction time available.
        i.e. final_time = final initial_time + maximum lead_time
    coords : dict
        Mapping of coordinate names to their corresponding coordinate values.
        Describes the coordinates associated with the dataset's dimensions.
    dims : tuple[str, ...]
        Names of the dataset dimensions, in their original order.
    time_coords_type : TimeTypes = Literal["datetime", "cftime"]
        Type of temporal coordinates used by the dataset, is in
        ["datetime", "cftime"]
    init_time_freq : TimeFrequency = Literal["day", "month", "year"]    
        Temporal resolution of the dataset's initialization times, must be 
        in ["day", "month", "year"]. This is used to determine the temporal 
        resolution of the dataset, which can be Used for calendar-aware time handling
        and temporal alignment.
    """

    sizes: dict | None
    start_time: xr.DataArray | np.ndarray | str | int | None
    final_time: xr.DataArray | np.ndarray | str | int | None
    coords: dict
    dims: tuple[str, ...]
    time_coords_type: TimeTypes | None
    init_time_freq: TimeFrequency | None


class DataConfigABC(abc.ABC):

    """
    Abstract base class for DataConfig objects, providing a standardized interface for handling
    xarray datasets in the CCCma_PPP framework. 

    Defines a common interface for model and observational datasets, including
    temporal resolution, ensemble selection, temporal and spatial metadata,
    preprocessing, and data access. Requires a coordinate matching each dimension,
    requires time coordinate to be either a datetime or cftime type, 
    and requires the initialization time coordinate to be present as specified for the pipeline.

    Subclasses must define the dataset type (`TYPE`), allowed dimensions,
    required dimensions, and the configuration attributes needed to resolve
    their data sources.

    During initialization, the class resolves the input files, configures
    the preprocessing pipeline, and extracts dataset metadata. The underlying
    xarray dataset is opened separately through `open_xarray_data`.

    Attributes
    ----------
    paths : str
        Path or path specification identifying the input data files.
    names : list[str]
        Names of the variables to load from the input files.
    preprocessing_pipeline : PreprocessingPipeline
        Pipeline containing the preprocessing operations applied to the
        dataset. Can be fitted using selected training data or restored
        from a previously saved pipeline.
    realization_list : list | None
        List of ensemble realizations to select. If None, 
        all available realizations are loaded. 
    ensemble_mean : bool | None
        Whether to average ensemble realizations during data loading.
    concat_dim : str | None
        Dimension along which data from multiple input files are
        concatenated.
    file_type : str
        File format of the input data.
    rename_dict : dict
        Mapping of original dimension, coordinate, or variable names
        to their standardized names used by the pipeline.

    Notes
    -----
    Dataset metadata is stored as infoclass object, in ``info``.

    The underlying dataset is stored in `data` and remains unopened until
    `open_xarray_data` is called.

    Subclasses must implement `TYPE`, `_allowed_dims`, and `_required_dims`.
    """


    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline
    realization_list: list | None
    ensemble_mean: bool | None
    concat_dim: str | None
    file_type: str
    rename_dict: dict | None
    drop_vars_list: list | None

    init_time_dim: ClassVar[str] = init_time_dim
    lead_time_dim: ClassVar[str] = lead_time_dim
    realization_dim: ClassVar[str] = realization_dim
    lead_time_resolution: ClassVar[str] = lead_time_resolution
    supported_NN_dimensions: ClassVar[tuple] = supported_NN_dimensions_sorted

    def __init__(self):

        """
        Initialize the data configuration and resolve its metadata.

        Validates the presence of the preprocessing pipeline, configures
        ensemble-selection checks, assigns the preprocessing pipeline name,
        resolves the input data sources, and extracts dataset metadata.

        The underlying xarray dataset is not retained in memory during
        initialization and must be opened separately.

        Raises
        ------
        AttributeError
            If the subclass does not define `preprocessing_pipeline`.
        """

        if not hasattr(self, "preprocessing_pipeline"):
            raise AttributeError(
                f"{type(self).__name__} must define preprocessing_pipeline"
            )

        self.data = None
        self._check_ensemble = False

        if self.realization_list is not None:
            self._check_ensemble = True

        self.preprocessing_pipeline.set_name(self.TYPE, self.rename_dict)

        _resolve_data(self)
        self.info = _get_ds_info(self)

    @property
    @abc.abstractmethod
    def TYPE(self) -> str:
        """
        Return the data source type identifier, e.g. model data or observations.
        Must be implemented by subclasses to specify the dataset source.

        Used to distinguish between different dataset sources and
        to construct the preprocessing pipeline name.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Return the set of dimensions permitted for this data source type.

        Subclasses must implement this method to define which dimensions
        are supported when validating and resolving input datasets.

        Returns
        -------
        frozenset[str]
            Names of all dimensions permitted for this dataset type.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Return the set of dimensions required for this data source type.

        Subclasses must implement this method to specify the dimensions
        that must be present in every valid input dataset.

        Returns
        -------
        frozenset[str]
            Names of dimensions required for this dataset type.
        """
        pass

    @final
    def fit_preprocessor_pipeline(
        self,
        selection: dict,
        mask: bool = False,
        save: bool = True,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ):

        """
        Fit the preprocessing pipeline using a selected subset of the input data.

        Loads the requested variables and data subset, optionally constructs
        a training mask based on initialization and lead times, and fits
        the configured preprocessing pipeline.

        The fitted pipeline can optionally be saved for reuse during
        subsequent training or inference.

        Parameters
        ----------
        selection : dict
            Mapping of dimension or coordinate names to the values or
            ranges used to select the data for fitting. Typically used
            to restrict preprocessing statistics to the training period only.
        mask : bool, default False
            Whether to construct and apply a training mask using the
            dataset's initialization and lead times. USed to prevent data 
            leakage into future times when fitting preprocessing statistics.
        save : bool, default True
            Whether to save the fitted preprocessing pipeline.
        save_path : Path | str | None, default None
            Directory in which to save the fitted pipeline. If None,
            the preprocessing pipeline determines the default location.
        save_name : str | None, default None
            Filename or identifier used when saving the fitted pipeline.
            If None, the preprocessing pipeline determines the default name.

        Notes
        -----
        Preprocessing statistics are estimated from the selected data,
        subject to the optional training mask.

        The temporary dataset is loaded into memory before fitting and
        closed after the fitting operation completes.
        """

        _base = _load_xarray_data(
            self.list_paths,
            names=self.names,
            concat_dim=self.concat_dim,
            selection=selection,
            ensemble_mean=self.ensemble_mean,
            rename_dict=self.rename_dict,
            drop_vars_list=self.drop_vars_list,
            resolution_to_standardize_time=self.lead_time_resolution,
        )

        _mask = (
            _create_train_mask(_base[self.init_time_dim], _base[self.lead_time_dim])
            if (mask and
            self.init_time_dim in _base.dims)
            else None
        )

        self.preprocessing_pipeline.fit(
            base_data=_base.load(),
            mask=_mask,
            save=save,
            save_path=save_path,
            save_name=save_name,
        )

        _base.close()
        del _base, _mask
        gc.collect()

    @final
    def load_preprocessor_pipeline(self, load_dir: Path | str | None = None):

        """
        Restore a previously fitted preprocessing pipeline from disk.

        Loads the serialized preprocessing pipeline associated with this
        data configuration and verifies that the restored pipeline
        has been fitted.

        Parameters
        ----------
        load_dir : Path | str | None, default None
            Directory containing the saved preprocessing pipeline.
            If None, defaults to the `preprocessing_pipeline` directory
            under `RuntimeContext.GLOBAL_EXP_DIR`.

            The expected filename is constructed from the preprocessing
            pipeline name using the suffix `_preprocessing_pipeline.joblib`.

        Raises
        ------
        RuntimeError
            If the restored preprocessing pipeline is not fitted.
        """

        if load_dir is None:
            load_dir = Path(RuntimeContext.GLOBAL_EXP_DIR) / "preprocessing_pipeline"

        load_dir = (
            Path(load_dir)
            / f"{self.preprocessing_pipeline.name}_preprocessing_pipeline.joblib"
        )

        self.preprocessing_pipeline.load_from_memory(
            Path(load_dir),
        )

        if not self.preprocessing_pipeline.fitted:
            raise RuntimeError(
                f"the loaded preprocessor for {self.preprocessing_pipeline.name} is not fitted!"
            )

    @final
    def open_xarray_data(
        self, load: bool = False, add_time_auxiliary_coords: bool = False
    ):
        """
        Open the configured input files as an xarray dataset.

        Loads the requested variables, applies the configured ensemble
        selection and dimension renaming, and concatenates multiple files
        along the specified dimension.

        The (initialization-)time coordinate is standardized using the finer
        temporal resolution of the initialization_times and lead_times.
        The resulting dataset is stored in `self.data`.

        Parameters
        ----------
        load : bool, default False
            Whether to load the entire dataset into memory immediately.
            If False, data may remain lazily loaded, depending on the
            underlying xarray backend.
        add_time_auxiliary_coords : bool, default False
            Whether to generate additional integer temporal coordinates 
            for day, month, and year based on the initialization time coordinate. 

        Notes
        -----
        If realization coordinates are specified in the dataset metadata,
        only the corresponding ensemble realizations are selected.

        Calling this method again replaces the reference stored in
        `self.data`. Use `close_data` to explicitly close an existing
        dataset before reopening it.
        """

        self.data = _load_xarray_data(
            self.list_paths,
            names=self.names,
            ensemble_mean=self.ensemble_mean,
            selection={self.realization_dim: self.coords[self.realization_dim]}
            if self.coords.get(self.realization_dim) is not None
            else None,
            concat_dim=self.concat_dim,
            rename_dict=self.rename_dict,
            drop_vars_list=self.drop_vars_list,
            resolution_to_standardize_time=self.lead_time_resolution,
            add_time_auxiliary_coords=add_time_auxiliary_coords,
            load=load,
        )
        
        if self.init_time_dim in self.data:
            time = self.data.coords[self.init_time_dim]
            _validate_time_sequence(time)

        return self

    @final
    def isel(
        self,
        indexers: dict[str, Any] | None = None,
        **indexers_kwargs: Any,
    ):

        """
        Wrapper for xarray Dataset.isel() method, applies 
        the preprocessing pipeline to the selected data before returning it.

        notes
        -----
        The underlying xarray dataset must be opened first using `open_xarray_data`.
        """

        self._check_opened()

        ds = self.data.isel(indexers=indexers, **indexers_kwargs)

        return self.preprocessing_pipeline.transform(ds)

    @final
    @property
    def coords(self) -> Mapping[str, int]:
        """
        Return the coordinate metadata of the configured data source.
        """
        return self.info.coords

    @final
    @property
    def sizes(self) -> Mapping[str, int]:
        """
        Return the size metadata of the configured data source.
        """
        return self.info.sizes

    @final
    @property
    def dims(self) -> tuple[str, ...] | Mapping[str, int]:
        """
        Return the dimension metadata of the configured data source.
        """
        return self.info.dims

    @final
    @property
    def init_time_frequency(self) -> TimeFrequency:
        """
        Return the initialization time frequency of the configured data source.
        """
        return self.info.init_time_freq

    @final
    @property
    def indexes(self) -> Mapping[str, pd.Index]:
        """
        Wrapper for xarray Dataset.indexes property.

        notes
        -----
        The underlying xarray dataset must be opened first using `open_xarray_data`.
        """
        self._check_opened()
        return self.data.indexes

    @final
    def _check_opened(self):
        """
        Verify that the underlying xarray dataset is currently open.
        """
        if self.data is None:
            raise ValueError(
                "No data is currently opened. "
                "Make sure '_open_xarray_data' is called first."
            )

    @final
    def close_data(self):
        """Close the xarray dataset and release OS file handles."""
        if self.data is not None:
            self.data.close()
            self.data = None

    def __del__(self):

        """
        Attempt to close the underlying dataset when the instance is destroyed.

        Calls `close_data` to release resources associated with the
        opened xarray dataset.

        Notes
        -----
        The timing of object destruction is not guaranteed. Explicitly
        calling `close_data` is recommended when the dataset is no
        longer needed.
        """

        self.close_data()


def _resolve_data(dataconfig: DataConfigABC, _do_checks: bool = True) -> None:

    """
    Discover and optionally validate input data files for a DataConfig.

    Finds files matching the configured file pattern and validates their
    dimensions, coordinates, variables, and initialization-time information.
    Dimension and coordinate names are standardized using the configured
    renaming dictionary before validation.

    Required dimensions must be present in each individual file, except
    for the configured concatenation dimension, which may instead exist
    as a coordinate. The concatenation coordinate must be present in
    every file, even when it is not a dimension. 

    Requires a coordinate matching each exisiting dimension.

    Resolved file paths are stored in `dataconfig.list_paths`.

    Parameters
    ----------
    dataconfig : DataConfigABC
        Data source configuration defining the input paths, file pattern,
        dimension requirements, coordinate names, and requested variables.
    _do_checks : bool, default True
        Whether to validate each discovered file. If False, only file
        discovery and existence checks are performed.

    Raises
    ------
    FileNotFoundError
        If the configured input path does not exist or no files matching
        the configured file pattern are found.
    ValueError
        If any discovered file fails validation, including when:

        - Required dimensions are missing, except when the only missing
        dimension matches `concat_dim` and exists as a coordinate.
        - The configured concatenation coordinate is missing.
        - Dataset dimensions are not permitted by the configuration.
        - None of the supported neural-network dimensions are present.
        - Dataset dimensions lack corresponding coordinates.
        - Initialization-time coordinates fail temporal sequence validation.
        - Ensemble selection is requested but the realization dimension
        is absent.
        - Requested data variables are missing.

    Notes
    -----
    Files are discovered using a glob pattern constructed from
    `dataconfig.paths` and `dataconfig.file_type`.

    Each file is opened independently using xarray and closed after
    validation. The complete dataset is not loaded into memory.

    The concatenation coordinate may be either a dimension coordinate
    or a non-dimension coordinate, allowing individual files to be
    concatenated along a dimension introduced during loading.

    Initialization-time sequence validation is performed only when the
    configured initialization-time coordinate is present.

    Validation is performed independently for each file. Consistency
    of coordinate values, spatial grids, and temporal continuity across
    files is not explicitly checked.

    This function modifies `dataconfig.list_paths` in place and returns
    None.
    """

    if not Path(dataconfig.paths).exists():
        raise FileNotFoundError(
            "The following file does not exist:\n" + str(dataconfig.paths)
        )

    list_paths = glob.glob(str(Path(dataconfig.paths).joinpath(dataconfig.file_type)))

    if len(list_paths) == 0:
        raise FileNotFoundError(
            f"The following file does is empty for {dataconfig.file_type} file type :\n"
            + str(dataconfig.paths)
        )

    if _do_checks:
        for p in list_paths:
            with xr.open_dataset(Path(p)) as ds:
                if dataconfig.rename_dict is not None:
                    ds = ds.rename(dataconfig.rename_dict)

                if dataconfig.drop_vars_list is not None:
                    ds = ds.drop_vars(dataconfig.drop_vars_list, errors="ignore")

                ds_dims = set(ds.dims)
                   
                invalid = dataconfig._required_dims() - ds_dims
                if invalid:
                    valid_concat = (
                        dataconfig.concat_dim is not None
                        and invalid == {dataconfig.concat_dim}
                        and dataconfig.concat_dim in ds.coords
                    )
                    if not valid_concat:
                        raise ValueError(
                            f"{dataconfig.TYPE} data is missing required dimensions: {sorted(invalid)}. "
                            f"A missing dimension is only allowed if it matches the configured "
                            f"concat_dim ({dataconfig.concat_dim!r}) and exists as a coordinate "
                            f"in the individual data file. File: {p}"
                        )

                invalid = ds_dims - dataconfig._allowed_dims()
                if invalid:
                    raise ValueError(
                        f"invalid data dimensions {list(ds.dims)} for {dataconfig.TYPE} data. Must be a subset ot {sorted(dataconfig._allowed_dims())} for {p}"
                    )

                if not set(dataconfig.supported_NN_dimensions).intersection(ds_dims):
                    raise ValueError(
                        f'"None of the supported NN dimensions exist in {p}'
                    )

                invalid = ds_dims - set(ds.coords.keys())
                if invalid:
                    raise ValueError(
                        f'"coordinates for {list(ds.dims)} does not exist. Available coords: {list(ds.coords.keys())} for {p}'
                    )

                invalid = (
                    dataconfig.concat_dim is not None 
                    and dataconfig.concat_dim not in ds.coords
                )
                if invalid:
                        raise ValueError(
                            f"Requested concat dimension {dataconfig.concat_dim} "
                            f"is not a valid data coordinate for {p}"
                        )

                _check_init_time = (
                    dataconfig.init_time_dim in ds_dims 
                    or dataconfig.init_time_dim in ds.coords
                )
                if _check_init_time:
                    time = ds.coords[dataconfig.init_time_dim]
                    _validate_time_sequence(time)

                if dataconfig._check_ensemble:
                    if dataconfig.realization_dim not in ds.dims:
                        raise ValueError(
                            f"Cannot select realization_list as {dataconfig.realization_dim} dim does not exist in {p}"
                        )

                missing = [
                    name for name in dataconfig.names if name not in ds.data_vars
                ]
                if missing:
                    raise ValueError(f"{p} is missing variables: {missing}")

    dataconfig.list_paths = list_paths


def _get_ds_info(dataconfig: DataConfigABC) -> infoclass:

    """
    Extract information about xarray data as an infoclass instance.

    Opens the input files associated with the dataset configuration,
    applies the requested variable selection, coordinate renaming,
    and optional ensemble selection, and extracts the metadata required
    by the pipeline.

    The dataset is closed after metadata extraction, and the collected
    information is returned as an `infoclass` instance.

    Parameters
    ----------
    dataconfig : DataConfigABC
        Data source configuration specifying the input files, requested
        variables, ensemble realizations, concatenation dimension,
        coordinate renaming rules, and standardized dimension names.

    Returns
    -------
    infoclass
        Instance containing the extracted metadata.

    Notes
    -----
    Previously resolved file paths are used when available.
    Otherwise, input files are discovered using the configured
    directory and file type.

    If `realization_list` is specified, metadata is extracted from
    the **selected ensemble realizations** rather than the complete
    ensemble.

    Initialization-time frequency is inferred from the dataset's
    time index rather than specified explicitly.

    The underlying dataset is closed after metadata extraction.
    No dataset is retained by this function.
    """

    init_time_dim = dataconfig.init_time_dim
    lead_time_dim = dataconfig.lead_time_dim

    if getattr(dataconfig, "list_paths", None) is None:
        list_paths = glob.glob(
            str(Path(dataconfig.paths).joinpath(dataconfig.file_type))
        )
    else:
        list_paths = dataconfig.list_paths

    ds = _load_xarray_data(
        list_paths,
        names=dataconfig.names,
        selection={dataconfig.realization_dim: dataconfig.realization_list}
        if dataconfig.realization_list is not None
        else None,
        concat_dim=dataconfig.concat_dim,
        rename_dict=dataconfig.rename_dict,
        drop_vars_list=dataconfig.drop_vars_list,
        resolution_to_standardize_time=dataconfig.lead_time_resolution,
    )

    if dataconfig.realization_list is not None:
        ds = ds.sel({dataconfig.realization_dim: dataconfig.realization_list})

    if init_time_dim in ds.dims:

        start_time, final_time = (
            ds[init_time_dim].min().values,
            ds[init_time_dim].max().values,
        )

        time_coords_type = get_time_representation(ds[init_time_dim])
        time_freq = infer_time_resolution(ds.coords[init_time_dim].to_index())

    else:
        start_time = None
        final_time = None
        time_coords_type = None
        time_freq = None

    sizes = {
        dim: dict(ds.sizes).get(dim)
        for dim in dict(ds.sizes).keys()
        if (
            dim in (init_time_dim, lead_time_dim)
            or dim in (dataconfig.realization_dim,)
        )
    }
    if not sizes:
        sizes = None

    coords = {dim: dict(ds.coords).get(dim) for dim in ds.coords}
    dims = ds.dims

    ds.close()
    del ds

    return infoclass(
        start_time=start_time,
        final_time=final_time,
        sizes=sizes,
        coords=coords,
        dims=dims,
        time_coords_type=time_coords_type,
        init_time_freq=time_freq,
    )
