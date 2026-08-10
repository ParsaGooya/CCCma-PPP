import abc
from typing import final, ClassVar
from pathlib import Path
import gc
import glob
import xarray as xr
import numpy as np
import dataclasses

from cccma_ppp.preprocessing.preprocessing import PreprocessingPipeline
from cccma_ppp.configs import (
    required_sample_dimensions,
    realization_dim,
    supported_NN_dimensions_sorted,
)

from cccma_ppp.data_modules.utils import (
    _load_xarray_data,
    _create_train_mask,
    _validate_time_sequence,
    infer_time_resolution,
    get_time_representation,
    TimeTypes,
    TimeFrequency
)
from cccma_ppp.generic.runtime import RuntimeContext



init_time_dim, lead_time_dim = required_sample_dimensions
@dataclasses.dataclass
class infoclass:
    """
    Container for dataset metadata.

    Parameters
    ----------
    sizes : dict or None
        Sizes of non-spatial dataset dimensions.
    start_time : xr.DataArray or np.ndarray or str or int or None
        Earliest available time.
    final_time : xr.DataArray or np.ndarray or str or int or None
        Latest available time.
    coords : dict
        Spatial and ensembles coordinates.
    """

    sizes: dict | None
    start_time: xr.DataArray | np.ndarray | str | int | None
    final_time: xr.DataArray | np.ndarray | str | int | None
    coords: dict
    time_coords_type: TimeTypes 
    init_time_freq: TimeFrequency


class DataConfigABC(abc.ABC):
    """
    Abstract base class for dataset configuration.
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
    supported_NN_dimensions: ClassVar[tuple] = supported_NN_dimensions_sorted


    def __init__(self):
        """
        Initialize data configuration.

        Ensures preprocessing pipeline exists and assigns its name.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If preprocessing_pipeline is not defined.
        """
        if not hasattr(self, "preprocessing_pipeline"):
            raise AttributeError(
                f"{type(self).__name__} must define preprocessing_pipeline"
            )

        self._check_ensemble = False
        if self.realization_list is not None:
            self._check_ensemble = True

        self.preprocessing_pipeline.set_name(self.TYPE)

        self._resolve_data()
        self.info = self._get_ds_info()

    @property
    @abc.abstractmethod
    def TYPE(self) -> str:
        """
        Type identifier for dataset.

        Returns
        -------
        str
        """

        pass

    @classmethod
    @abc.abstractmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Allowed dataset dimensions.

        Returns
        -------
        frozenset of str
            Set of allowed dataset dimension names.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Required dataset dimensions.

        Returns
        -------
        frozenset of str
        """
        pass

    @final
    def _resolve_data(self):
        """
        Validate dataset files and structure.

        Returns
        -------
        None
        """
        _resolve_data(self)

    @final
    def _get_ds_info(self):
        """
        Extract dataset metadata.

        Returns
        -------
        infoclass
        """
        return _get_ds_info(self)

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
        Fit preprocessing pipeline on dataset.

        Parameters
        ----------
        selection : dict
            Subset selection for dataset.
        mask : bool, optional
            Whether to apply training mask.
        save : bool, optional
            Whether to save pipeline.
        save_path : pathlib.Path or str or None, optional
        save_name : str or None, optional

        Returns
        -------
        None
        """
        
        _base = _load_xarray_data(
            self.list_paths,
            names=self.names,
            concat_dim=self.concat_dim,
            selection=selection,
            ensemble_mean=self.ensemble_mean,
            rename_dict=self.rename_dict,
        )

        _mask = _create_train_mask(_base[self.init_time_dim], _base[self.lead_time_dim]) if mask else None

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
        Load fitted preprocessing pipeline.

        Parameters
        ----------
        load_dir : pathlib.Path or str or None

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If loaded pipeline is not fitted.
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


def _resolve_data(dataconfig: DataConfigABC, 
                  _do_checks: bool = True) -> None:
    """
    Validate dataset files and dimensions.

    Parameters
    ----------
    dataconfig : DataConfigABC

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data files do not exist.
    ValueError
        If dataset dimensions or variables are invalid.
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
    Extract dataset metadata information.

    Parameters
    ----------
    dataconfig : DataConfigABC

    Returns
    -------
    infoclass
        Metadata describing dataset dimensions and coordinates.
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
        concat_dim=dataconfig.concat_dim,
        rename_dict=dataconfig.rename_dict,
    )

    if dataconfig.realization_list is not None:
        ds = ds.sel({dataconfig.realization_dim : dataconfig.realization_list})

    if init_time_dim in ds.dims:
        start_time, final_time = ds[init_time_dim].min().values, ds[init_time_dim].max().values
    else:
        start_time = final_time = None

    sizes = {
        dim: dict(ds.sizes).get(dim)
        for dim in dict(ds.sizes).keys()
        if (dim in (init_time_dim, lead_time_dim) or dim in (dataconfig.realization_dim,))
    }
    if not sizes:
        sizes = None

    coords = {dim: dict(ds.coords).get(dim) for dim in ds.coords}

    time_coords_type = get_time_representation(ds[init_time_dim])

    time_freq = infer_time_resolution(ds.coords[init_time_dim].to_index())

    ds.close()
    del ds

    return infoclass(
        start_time=start_time, 
        final_time=final_time, 
        sizes=sizes, 
        coords=coords, 
        time_coords_type=time_coords_type,
        init_time_freq=time_freq
    )



