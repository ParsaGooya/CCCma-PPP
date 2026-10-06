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
    Container for metadata describing a configured dataset.

    Parameters
    ----------
    sizes : dict | None
        Sizes of tracked dataset dimensions, or ``None`` when none are present.
    start_time : xr.DataArray | np.ndarray | str | int | None
        Earliest initialization time in the dataset, or ``None`` when the
        initialization time is not a dimension.
    final_time : xr.DataArray | np.ndarray | str | int | None
        Latest initialization time in the dataset, or ``None`` when the
        initialization time is not a dimension.
    coords : dict
        Mapping of coordinate names to their coordinate values.
    dims : tuple[str, ...]
        Dimensions present in the dataset.
    time_coords_type : TimeTypes
        Representation used by the initialization-time coordinate.
    init_time_freq : TimeFrequency
        Temporal resolution inferred from the initialization-time coordinate.
    """

    sizes: dict | None
    start_time: xr.DataArray | np.ndarray | str | int | None
    final_time: xr.DataArray | np.ndarray | str | int | None
    coords: dict
    dims: tuple[str, ...]
    time_coords_type: TimeTypes
    init_time_freq: TimeFrequency


class DataConfigABC(abc.ABC):
    """
    Abstract base class for dataset configurations.

    Defines common data discovery, validation, preprocessing, loading, selection,
    and metadata operations for configured datasets.

    Attributes
    ----------
    paths : str
        Directory containing the dataset files.
    names : list[str]
        Names of data variables to load.
    preprocessing_pipeline : PreprocessingPipeline
        Pipeline used to preprocess loaded data.
    realization_list : list | None
        Realizations to select from ensemble data. If ``None``, no explicit
        realization subset is selected.
    ensemble_mean : bool | None
        Whether ensemble data are reduced to their ensemble mean.
    concat_dim : str
        Dimension along which data from multiple files are concatenated.
    file_type : str
        File pattern used to discover data files within ``paths``.
    rename_dict : dict
        Mapping used to rename dataset variables, dimensions, or coordinates.
    """

    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline
    realization_list: list | None
    ensemble_mean: bool | None
    concat_dim: str
    file_type: str
    rename_dict: dict

    init_time_dim: ClassVar[str] = init_time_dim
    lead_time_dim: ClassVar[str] = lead_time_dim
    realization_dim: ClassVar[str] = realization_dim
    lead_time_resolution: ClassVar[str] = lead_time_resolution
    supported_NN_dimensions: ClassVar[tuple] = supported_NN_dimensions_sorted

    def __init__(self):
        """
        Initialize a data configuration.

        Validates that a preprocessing pipeline is defined, configures ensemble
        handling, associates the preprocessing pipeline with the data type and
        renaming rules, resolves and validates the configured data files, and
        extracts dataset metadata.

        Raises
        ------
        AttributeError
            If ``preprocessing_pipeline`` is not defined by the configuration.
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
        Return the configured data type.

        Returns
        -------
        str
            Identifier describing the type of data represented by the configuration.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Return the dimensions permitted for this data type.

        Returns
        -------
        frozenset[str]
            Names of all dimensions allowed in configured datasets.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Return the dimensions required for this data type.

        Returns
        -------
        frozenset[str]
            Names of dimensions that must be present in configured datasets.
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
        Fit the preprocessing pipeline using a selected subset of the dataset.

        The configured variables are loaded according to ``selection`` and an
        optional training mask is constructed before fitting the preprocessing
        pipeline.

        Parameters
        ----------
        selection : dict
            Coordinate selections applied when loading data for fitting.
        mask : bool
            Whether to construct and provide a training mask to the preprocessing
            pipeline.
        save : bool
            Whether to save the fitted preprocessing pipeline.
        save_path : Path | str | None
            Directory in which to save the fitted preprocessing pipeline.
        save_name : str | None
            Name used when saving the fitted preprocessing pipeline.
        """
        _base = _load_xarray_data(
            self.list_paths,
            names=self.names,
            concat_dim=self.concat_dim,
            selection=selection,
            ensemble_mean=self.ensemble_mean,
            rename_dict=self.rename_dict,
        )

        _mask = (
            _create_train_mask(_base[self.init_time_dim], _base[self.lead_time_dim])
            if mask
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
        Load a fitted preprocessing pipeline from disk.

        When ``load_dir`` is not provided, the preprocessing pipeline is loaded from
        the current experiment's preprocessing directory.

        Parameters
        ----------
        load_dir : Path | str | None
            Directory containing the saved preprocessing pipeline. If ``None``, the
            current experiment preprocessing directory is used.

        Raises
        ------
        RuntimeError
            If the loaded preprocessing pipeline is not fitted.
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
        Open the configured dataset.

        Data variables, ensemble selections, renaming rules, and temporal resolution
        information are applied when loading the dataset.

        Parameters
        ----------
        load : bool
            Whether to load the opened data into memory.
        add_time_auxiliary_coords : bool
            Whether to add auxiliary temporal coordinates while loading the data.

        Returns
        -------
        DataConfigABC
            This configuration instance with the opened dataset assigned to
            ``data``.
        """
        time_frequency_to_resolve = min(
            self.info.init_time_freq,
            self.info.lead_time_resolution,
            key=TEMPORAL_ORDER.get,
        )

        self.data = _load_xarray_data(
            self.list_paths,
            names=self.names,
            ensemble_mean=self.ensemble_mean,
            selection={self.realization_dim: self.coords[self.realization_dim]}
            if self.coords.get(self.realization_dim) is not None
            else None,
            concat_dim=self.concat_dim,
            rename_dict=self.rename_dict,
            resolved_init_time_frequency=time_frequency_to_resolve,
            add_time_auxiliary_coords=add_time_auxiliary_coords,
            load=load,
        )

        return self

    @final
    def isel(
        self,
        indexers: dict[str, Any] | None = None,
        **indexers_kwargs: Any,
    ):
        """
        Select data by integer position and apply preprocessing.

        Parameters
        ----------
        indexers : dict[str, Any] | None
            Mapping from dimension names to positional indexers.
        **indexers_kwargs : Any
            Additional dimension indexers passed to the underlying xarray
            ``isel`` operation.

        Returns
        -------
        Any
            Selected data after applying the configured preprocessing pipeline.
        """
        self._check_opened()

        ds = self.data.isel(indexers=indexers, **indexers_kwargs)

        return self.preprocessing_pipeline.transform(ds)

    @final
    @property
    def coords(self) -> Mapping[str, int]:
        """
        Return the dataset coordinates recorded during configuration.

        Returns
        -------
        Mapping[str, int]
            Mapping containing the available dataset coordinates.
        """
        return self.info.coords

    @final
    @property
    def sizes(self) -> Mapping[str, int]:
        """
        Return the tracked dataset dimension sizes.

        Returns
        -------
        Mapping[str, int] | None
            Mapping from tracked dimension names to their sizes, or ``None`` when no
            tracked dimensions are present.
        """
        return self.info.sizes

    @final
    @property
    def dims(self) -> tuple[str, ...] | Mapping[str, int]:
        """
        Return the dimensions present in the configured dataset.

        Returns
        -------
        tuple[str, ...] | Mapping[str, int]
            Dataset dimension information recorded during configuration.
        """
        return self.info.dims

    @final
    @property
    def init_time_frequency(self) -> str:
        """
        Return the temporal resolution of the initialization-time coordinate.

        Returns
        -------
        TimeFrequency
            Temporal resolution inferred from the initialization times.
        """
        return self.info.init_time_freq

    @final
    @property
    def indexes(self) -> Mapping[str, pd.Index]:
        """
        Return indexes from the currently opened dataset.

        Returns
        -------
        Mapping[str, pd.Index]
            Mapping from indexed dimension names to their pandas indexes.
        """
        self._check_opened()
        return self.data.indexes

    @final
    def _check_opened(self):
        """
        Verify that dataset data have been opened.

        Raises
        ------
        ValueError
            If no dataset is currently assigned to ``data``.
        """
        if self.data is None:
            raise ValueError(
                "No data is currently opened. "
                "Make sure '_open_xarray_data' is called first."
            )

    @final
    def close_data(self):
        """
        Close the currently opened dataset and release its file handles.

        If no dataset is open, this method has no effect.
        """
        if self.data is not None:
            self.data.close()
            self.data = None

    def __del__(self):
        self.close_data()


def _resolve_data(dataconfig: DataConfigABC, _do_checks: bool = True) -> None:
    """
    Discover and optionally validate files for a data configuration.

    Matching files are located using the configured path and file pattern. When
    validation is enabled, each dataset is checked for required and allowed
    dimensions, coordinates, requested variables, ensemble dimensions, supported
    neural-network dimensions, and a valid initialization-time sequence.

    Parameters
    ----------
    dataconfig : DataConfigABC
        Data configuration whose files are discovered and validated.
    _do_checks : bool
        Whether to validate the structure and contents of each discovered file.

    Raises
    ------
    FileNotFoundError
        If the configured data path does not exist or no files match the
        configured file pattern.
    ValueError
        If a dataset is missing required dimensions, coordinates, requested
        variables, or ensemble dimensions; contains unsupported dimensions; has
        no supported neural-network dimension; or contains an invalid
        initialization-time sequence.
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

                ds_dims = set(ds.dims)

                if dataconfig.init_time_dim not in ds_dims:
                    if dataconfig.init_time_dim not in ds.coords:
                        raise ValueError(
                            f"The required initialization time ({dataconfig.init_time_dim} ) must be a dimension or at least a coordinate of individual data "
                            "file which will be a dimension after concatenation."
                        )

                invalid = dataconfig._required_dims() - ds_dims
                if invalid:
                    raise ValueError(
                        f"{dataconfig.TYPE} data must have {sorted(dataconfig._required_dims())} dimensions. Current dims : {list(ds.dims)} for {p}"
                    )

                if dataconfig._check_ensemble:
                    if dataconfig.realization_dim not in ds.dims:
                        raise ValueError(
                            f"Cannot select realization_list as {dataconfig.realization_dim} dim does not exist in {p}"
                        )

                invalid = ds_dims - dataconfig._allowed_dims()
                if invalid:
                    raise ValueError(
                        f"invalid data dimensions {list(ds.dims)} for {dataconfig.TYPE} data. Must be a subset ot {sorted(dataconfig._allowed_dims())} for {p}"
                    )
                invalid = ds_dims - set(ds.coords.keys())
                if invalid:
                    raise ValueError(
                        f'"coordinates for {list(ds.dims)} does not exist. Available coords: {list(ds.coords.keys())} for {p}'
                    )

                if not set(dataconfig.supported_NN_dimensions).intersection(ds_dims):
                    raise ValueError(
                        f'"None of the supported NN dimensions exist in {p}'
                    )

                missing = [
                    name for name in dataconfig.names if name not in ds.data_vars
                ]
                if missing:
                    raise ValueError(f"{p} is missing variables: {missing}")

                time = ds.coords[dataconfig.init_time_dim]
                _validate_time_sequence(time)

                ds.close()
                gc.collect()

    dataconfig.list_paths = list_paths


def _get_ds_info(dataconfig: DataConfigABC) -> infoclass:
    """
    Extract metadata from a configured dataset.

    The configured files are opened together and inspected to determine temporal
    coverage, tracked dimension sizes, coordinates, dimensions, time
    representation, and initialization-time resolution.

    Parameters
    ----------
    dataconfig : DataConfigABC
        Data configuration describing the files and selections to inspect.

    Returns
    -------
    infoclass
        Dataset metadata containing temporal coverage, dimension sizes,
        coordinates, dimensions, time representation, and initialization-time
        frequency.
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
    )

    if dataconfig.realization_list is not None:
        ds = ds.sel({dataconfig.realization_dim: dataconfig.realization_list})

    if init_time_dim in ds.dims:
        start_time, final_time = (
            ds[init_time_dim].min().values,
            ds[init_time_dim].max().values,
        )
    else:
        start_time = final_time = None

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

    time_coords_type = get_time_representation(ds[init_time_dim])

    time_freq = infer_time_resolution(ds.coords[init_time_dim].to_index())

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
