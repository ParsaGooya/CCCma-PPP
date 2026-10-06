import torch
from pathlib import Path
import logging
import dataclasses
import os
import time
from tqdm import tqdm
import numpy as np
import xarray as xr

from cccma_ppp.core.trainer import clear_memory

from cccma_ppp.data_modules.dataloader import Dataloader
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.generic.aggregator import RunningCovariance
from cccma_ppp.train.dataloader import TrainDataloaderConfig
from cccma_ppp.core.core_abc import moduleABC

from cccma_ppp.preprocessing.preprocessing import PreprocessingPipeline

from cccma_ppp.inference.predictors.deterministic import DeterministicPredictorConfig
from cccma_ppp.inference.predictors.cvae import cVAEPredictorConfig

from cccma_ppp.configs import required_sample_dimensions, realization_dim

init_time_dim, lead_time_dim = required_sample_dimensions


@dataclasses.dataclass
class WriterConfig:
    """
    Configuration for inference prediction and output writing.

    Parameters
    ----------
    predictor : DeterministicPredictorConfig | cVAEPredictorConfig | None
        Predictor configuration used to generate batch predictions. If ``None``,
        a deterministic predictor is used by default when prediction is enabled.
    num_output_sampling : int
        Number of output samples generated for each model input.
    get_trained_model_stats_from_validation : bool
        Whether training-variable statistics are extracted from the validation
        data instead of the training data.
    aggregate_only : bool
        Whether to skip prediction and aggregate previously saved temporary
        prediction files only.
    predict_only : bool
        Whether to generate temporary batch predictions without aggregating them
        into final NetCDF files.
    """

    predictor: DeterministicPredictorConfig | cVAEPredictorConfig | None = None
    num_output_sampling: int = 1
    get_trained_model_stats_from_validation: bool = False
    aggregate_only: bool = False
    predict_only: bool = False

    def __post_init__(self):
        """
        Validate the writer configuration.

        Raises
        ------
        ValueError
            If ``num_output_sampling`` is not positive.
        """

        if self.num_output_sampling <= 0:
            raise ValueError("num_output_sampling cannot be zero or negative.")

    def build(
        self,
        inference_data_loader: Dataloader,
        train_dataloader_config: TrainDataloaderConfig,
        module: moduleABC,
        post_processor: PreprocessingPipeline,
        output_dir: Path | str,
    ):
        """
        Build a prediction writer from this configuration.

        A default deterministic predictor is created when prediction is enabled and
        no predictor is explicitly configured. The predictor type is validated
        against the configured module type.

        Parameters
        ----------
        inference_data_loader : Dataloader
            Data loader providing samples for inference.
        train_dataloader_config : TrainDataloaderConfig
            Training data-loader configuration used when training statistics or
            metadata are required.
        module : moduleABC
            Trained module used to generate predictions.
        post_processor : PreprocessingPipeline
            Preprocessing pipeline used to transform predictions back to their output
            representation.
        output_dir : Path | str
            Directory in which temporary and aggregated predictions are stored.

        Returns
        -------
        Writer
            Configured inference writer.

        Raises
        ------
        RuntimeError
            If both ``aggregate_only`` and ``predict_only`` are enabled, or if the
            configured predictor type does not match the module type.
        """
        if self.aggregate_only and self.predict_only:
            raise RuntimeError(
                "You can run the writer in aggregate_only OR predict_only mode OR neither. "
                "In case of the former, temporary batch predictions must have been saved "
                "on the disc previously. In case of the later, batch predictions will be saved temporarily "
                "but will not be aggregated year by year."
            )
        if not self.aggregate_only:
            if self.predictor is None:
                self.predictor = DeterministicPredictorConfig()

            if self.predictor._type != module.config._type.lower():
                raise RuntimeError(
                    f"The provided selector config matches {self.predictor._type}"
                    f" selector but the module is {module.config._type.lower()}"
                )

        return Writer(
            config=self,
            inference_data_loader=inference_data_loader,
            train_dataloader_config=train_dataloader_config,
            module=module,
            post_processor=post_processor,
            output_dir=output_dir,
        )


class Writer:
    """
    Manage model inference and aggregation of prediction outputs.

    The writer generates temporary batch predictions, optionally extracts
    training-variable statistics, and aggregates temporary predictions into
    yearly NetCDF files.

    Parameters
    ----------
    config : WriterConfig
        Configuration controlling prediction, aggregation, and statistic
        extraction behavior.
    inference_data_loader : Dataloader
        Data loader providing samples for inference.
    train_dataloader_config : TrainDataloaderConfig
        Training data-loader configuration used when training statistics or
        metadata are required.
    module : moduleABC
        Trained module used to generate predictions.
    post_processor : PreprocessingPipeline
        Pipeline used to convert predictions back to their output representation.
    output_dir : Path | str
        Directory in which prediction outputs are stored.
    """

    def __init__(
        self,
        config: WriterConfig,
        inference_data_loader: Dataloader,
        train_dataloader_config: TrainDataloaderConfig,
        module: moduleABC,
        post_processor: PreprocessingPipeline,
        output_dir: Path | str,
    ):
        """
        Initialize an inference writer.

        Parameters
        ----------
        config : WriterConfig
            Configuration controlling prediction and aggregation behavior.
        inference_data_loader : Dataloader
            Data loader providing samples for inference.
        train_dataloader_config : TrainDataloaderConfig
            Training data-loader configuration used when additional training data are
            required.
        module : moduleABC
            Trained module used to generate predictions.
        post_processor : PreprocessingPipeline
            Pipeline used to convert predictions back to their output representation.
        output_dir : Path | str
            Directory in which prediction outputs are stored.
        """
        self.config = config
        self.module = module
        self.output_dir = Path(output_dir)
        self.InferenceLoader = inference_data_loader
        self.TrainLoaderConfig = train_dataloader_config
        self.post_processor = post_processor

        self._setup = False

    def setup_distributed(
        self,
        distributed: Distributed,
        logger: logging.Logger,
    ):
        """
        Configure the writer for distributed inference and output management.

        Distributed runtime information is recorded, temporary output storage is
        prepared, the predictor is constructed when required, and requested
        training-variable statistics are extracted.

        Parameters
        ----------
        distributed : Distributed
            Distributed runtime configuration containing device, rank, and process
            information.
        logger : logging.Logger
            Logger used for root-process inference messages.

        Raises
        ------
        RuntimeError
            If the module is not located on the writer device, or if
            ``aggregate_only`` mode is requested without previously saved temporary
            NetCDF prediction files.
        """
        self.logger = logger
        if self.logger is None:
            print("Logger is None. Print is used instead ... \n\n ")

        self.distributed = distributed
        self.device = distributed.device
        self.rank = distributed.rank
        self.local_rank = distributed.local_rank
        self.world_size = distributed.world_size
        self.is_distributed = distributed.distributed
        self.is_on_root = distributed.is_root()

        self.log_root(logging.INFO, "Setting up writer.")

        self.log_root(
            logging.INFO,
            f"rank={self.rank} | local_rank={self.local_rank} | world_size={self.world_size} | device={self.device}",
        )

        if self.raw_module._get_device() != self.device:
            raise RuntimeError(
                f"Module is on {self.raw_module._get_device()}, but trainer device is {self.device}"
            )

        self.temp_save_dir = Path(self.output_dir) / "_temp"

        if self.config.aggregate_only:
            if not (
                self.temp_save_dir.is_dir() and any(self.temp_save_dir.glob("*.nc"))
            ):
                raise RuntimeError(
                    "With aggregate_only writer mode, a _temp folder with at least "
                    "one ''.nc'' file must exist!"
                )

        if self.is_on_root:
            os.makedirs(self.temp_save_dir, exist_ok=True)

        if self.is_distributed:
            self.distributed.barrier()

        if not self.config.aggregate_only:
            self.predictor = self.config.predictor.build(
                self.module,
                self.distributed,
                self.output_dir,
                self.config.num_output_sampling,
            )

            if self.predictor.extract_training_vars:
                self.log_root(
                    logging.INFO,
                    "Running the model to extract training statistics. \n"
                    "This might take a few minutes...",
                )
                self._save_train_stats()

        self._setup = True
        self.log_root(logging.INFO, "Writer setup complete.")

    @property
    def train_stats_save_dir(self):
        """
        Return the path used to store training-variable statistics.

        Returns
        -------
        Path
            Path to the serialized training-variable statistics file.
        """
        return Path(self.output_dir) / "training_variable_stats.pt"

    def predict(self):
        """
        Run prediction and output aggregation.

        Depending on the writer configuration, this performs inference, aggregation,
        or both.

        Raises
        ------
        RuntimeError
            If distributed writer setup has not been completed before prediction.
        """
        if not self._setup:
            raise RuntimeError("Call setup_distributed() before predict().")
        if not self.config.aggregate_only:
            self.log_root(logging.INFO, "Starting Inference Loop...")
        else:
            self.log_root(logging.INFO, "Starting aggregation...")
        self.start_time = time.time()
        clear_memory()

        self._predict()

        time_elapsed = time.time() - self.start_time

        self.log_root(logging.INFO, f"Inference finished in {time_elapsed:.2f}s")

    def _predict(self):
        """
        Execute the configured inference and aggregation workflow.

        Prediction batches are processed when prediction is enabled. When latent
        outputs are requested, a training loader with metadata is used instead of the
        standard inference loader. Temporary predictions are subsequently aggregated
        unless ``predict_only`` is enabled.
        """
        self.module.eval()
        loader = self.InferenceLoader
        do_post_process = True

        if not self.config.aggregate_only and getattr(
            self.predictor, "save_latent", False
        ):
            loader = self.build_train_loader(return_metadata=True, shuffle=False)
            do_post_process = False

        with torch.inference_mode():
            if not self.config.aggregate_only:
                for batch in tqdm(
                    loader,
                    disable=not self.is_on_root,
                    desc="Inference",
                ):
                    batch = batch.to_device(self.device)

                    self.predictor._infer_on_batch(batch)

                del loader
                clear_memory()
            if not self.config.predict_only:
                self.aggregate_predictions_to_netcdf(do_post_process)

    def _save_train_stats(self):
        """
        Compute and save training-variable statistics when they are not cached.

        Statistics are collected from either the training or validation data loader,
        combined across distributed processes, and saved for subsequent inference
        use.
        """
        if not self.train_stats_save_dir.exists():
            loader = self.build_train_loader(
                from_validation=self.config.get_trained_model_stats_from_validation
            )

            stats = self.predictor.stats
            for batch in tqdm(
                loader,
                disable=not self.is_on_root,
                desc="Extracting training statistics",
            ):
                batch = batch.to_device(self.device)

                stats = self.predictor._infer_on_batch(
                    batch,
                    _getting_train_stats=True,
                )

            del loader
            clear_memory()
            self.aggregate_train_stats(stats)

        if self.is_distributed:
            self.distributed.barrier()

    def build_train_loader(
        self,
        from_validation: bool = False,
        return_metadata: bool = False,
        shuffle: bool | None = None,
    ):
        """
        Build a training or validation data loader for inference support operations.

        The training data-loader configuration is first initialized for the current
        distributed runtime using the saved preprocessing pipeline.

        Parameters
        ----------
        from_validation : bool
            Whether to build the validation loader instead of the training loader.
        return_metadata : bool
            Whether samples returned by the loader include associated metadata.
        shuffle : bool | None
            Whether samples are shuffled. If ``None``, the data-loader configuration
            determines the behavior.

        Returns
        -------
        Dataloader
            Configured training or validation data loader.
        """
        self.TrainLoaderConfig.setup_distributed(
            self.distributed,
            load_path=Path(RuntimeContext.GLOBAL_EXP_DIR) / "preprocessing_pipeline",
        )

        if from_validation:
            return self.TrainLoaderConfig.build_validation_loader(
                supress_error=False, return_metadata=return_metadata, shuffle=shuffle
            )
        else:
            return self.TrainLoaderConfig.build_train_loader(
                return_metadata=return_metadata, shuffle=shuffle
            )

    def aggregate_train_stats(self, stats: dict[str, RunningCovariance]):
        """
        Aggregate and save training-variable statistics.

        Running covariance accumulators are reduced across distributed processes.
        On the root process, finalized means and covariance matrices are written to
        the training statistics file.

        Parameters
        ----------
        stats : dict[str, RunningCovariance]
            Mapping from statistic names to running covariance accumulators.
        """
        for stat in stats.values():
            if stat.sum_x is not None:
                stat.distributed_reduce()

        if self.is_on_root:
            save_dict = {}

            for name, stat in stats.items():
                if stat.sum_x is None:
                    continue

                mean, cov = stat.finalize()
                save_dict[f"{name}_mean"] = mean
                save_dict[f"{name}_cov"] = cov

            torch.save(
                save_dict,
                self.train_stats_save_dir,
            )

        if self.is_distributed:
            self.distributed.barrier()

    def log_root(self, level: int, msg: str, *args):
        """
        Log a message only from the root process.

        If no logger is configured, the message is printed instead.

        Parameters
        ----------
        level : int
            Logging level associated with the message.
        msg : str
            Message to log.
        *args : Any
            Additional positional arguments passed to the configured logger.
        """
        if self.is_on_root:
            if self.logger is not None:
                self.logger.log(level, msg, *args)
            else:
                print(msg)

    @property
    def raw_module(self):
        """
        Return the underlying unwrapped model module.

        Returns
        -------
        moduleABC
            Module wrapped by distributed data parallel execution when applicable,
            or the original module otherwise.
        """
        if isinstance(self.module, torch.nn.parallel.DistributedDataParallel):
            return self.module.module
        return self.module

    def aggregate_predictions_to_netcdf(self, do_post_process: bool = True):
        """
        Aggregate temporary prediction files into yearly NetCDF outputs.

        Aggregation is performed only on the root process. Post-processing can be
        disabled when the temporary predictions are already in their desired output
        representation.

        Parameters
        ----------
        do_post_process : bool
            Whether to apply the configured inverse preprocessing pipeline during
            aggregation.
        """
        if self.is_distributed:
            self.distributed.barrier()

        post_processor = self.post_processor
        if not do_post_process:
            post_processor = None

        if self.is_on_root:
            aggregate_predictions(
                post_processor=post_processor,
                output_dir=self.output_dir,
                output_ensemble=(self.config.num_output_sampling > 1),
                logger_function=self.log_root,
            )

        if self.is_distributed:
            self.distributed.barrier()


def aggregate_predictions(
    post_processor: PreprocessingPipeline | None,
    output_dir: Path,
    output_ensemble: bool = False,
    logger_function: callable = None,
    cleanup_temp: bool = True,
    init_time_dim: str = init_time_dim,
    lead_time_dim: str = lead_time_dim,
    optional_sample_dimensions: tuple[str, ...] = (realization_dim,),
):
    """
    Aggregate temporary batch predictions into yearly NetCDF files.

    Temporary prediction files are combined by initialization time and lead time,
    sorted along sample coordinates, optionally inverse transformed, and written
    as one NetCDF file per initialization year. Duplicate lead times are removed
    before each initialization time is added to the yearly dataset.

    Parameters
    ----------
    post_processor : PreprocessingPipeline | None
        Pipeline used to convert predictions back to their output representation.
        If ``None``, prediction channels are converted directly into dataset
        variables without inverse preprocessing.
    output_dir : Path
        Directory containing the temporary prediction folder and receiving the
        aggregated NetCDF files.
    output_ensemble : bool
        Whether output filenames indicate that predictions contain an output
        ensemble.
    logger_function : callable | None
        Optional logging function used to report aggregation progress and saved
        files.
    cleanup_temp : bool
        Whether temporary prediction files and their directory are removed after
        successful aggregation.
    init_time_dim : str
        Name of the initialization-time coordinate.
    lead_time_dim : str
        Name of the lead-time coordinate.
    optional_sample_dimensions : tuple[str, ...]
        Additional sample dimensions to sort when present in the prediction data.

    Raises
    ------
    RuntimeError
        If no temporary prediction files are found, a temporary file lacks the
        initialization-time coordinate, no initialization times are available,
        or initialization or lead-time coordinates remain non-monotonic after
        aggregation.
    """
    output_dir = Path(output_dir)
    temp_save_dir = output_dir / "_temp"

    temp_files = sorted(temp_save_dir.glob("rank*_*.nc"))

    if not temp_files:
        raise RuntimeError(f"No temporary prediction files found in {temp_save_dir}")

    source = None
    nstds = "NA"
    all_times = None

    for path in temp_files:
        with xr.open_dataarray(path) as data:
            if init_time_dim not in data.coords:
                raise RuntimeError(
                    f"{path} does not contain the coordinate {init_time_dim!r}."
                )

            time_index = data.coords[init_time_dim].to_index()

            all_times = time_index if all_times is None else all_times.union(time_index)

            if source is None:
                source = data.attrs["source"]

            if nstds == "NA":
                nstds = data.attrs.get("nstds", None)

    if all_times is None or len(all_times) == 0:
        raise RuntimeError(
            "Temporary prediction files contain no initialization times."
        )

    all_times = all_times.sort_values()

    all_times_da = xr.DataArray(
        all_times,
        dims=(init_time_dim,),
        coords={init_time_dim: all_times},
    )

    initialization_years = np.asarray(all_times_da.dt.year.values)

    years = np.unique(initialization_years)

    if logger_function is not None:
        logger_function(
            logging.INFO,
            "Aggregating temporary prediction files year-by-year.",
        )

    sample_coords = (
        init_time_dim,
        lead_time_dim,
        *optional_sample_dimensions,
    )

    def _sort_sample_coords(
        data: xr.DataArray | xr.Dataset,
    ) -> xr.DataArray | xr.Dataset:
        """
        Sort data along all available sample coordinates.

        Parameters
        ----------
        data : xr.DataArray | xr.Dataset
            Data whose sample coordinates are to be sorted.

        Returns
        -------
        xr.DataArray | xr.Dataset
            Data sorted along each configured sample coordinate that is present.
        """
        for coord in sample_coords:
            if coord in data.coords:
                data = data.sortby(coord)

        return data

    for year in tqdm(years, desc="Saving years ..."):
        year_times = all_times[initialization_years == year]

        year_datasets = []

        for time in year_times:
            time_datasets = []

            for path in temp_files:
                with xr.open_dataarray(path) as data:
                    if init_time_dim not in data.coords:
                        raise RuntimeError(
                            f"{path} does not contain the coordinate {init_time_dim!r}."
                        )

                    available_times = data.coords[init_time_dim].to_index()

                    if time not in available_times:
                        continue

                    data_time_part = data.sel({init_time_dim: slice(time, time)})

                    for dim in sample_coords:
                        if dim != init_time_dim and dim in data_time_part.dims:
                            data_time_part = data_time_part.dropna(
                                dim=dim,
                                how="all",
                            )

                    data_time_part = _sort_sample_coords(data_time_part.load())

                    time_datasets.append(data_time_part)

            if not time_datasets:
                continue

            combined_time = xr.concat(
                time_datasets,
                dim=lead_time_dim,
                coords="minimal",
                compat="equals",
                join="exact",
            )

            lead_times = np.asarray(combined_time[lead_time_dim].values)

            _, unique_indices = np.unique(
                lead_times,
                return_index=True,
            )

            if len(unique_indices) != len(lead_times):
                combined_time = combined_time.isel(
                    {lead_time_dim: np.sort(unique_indices)}
                )

            combined_time = combined_time.sortby(lead_time_dim)

            if not combined_time.indexes[lead_time_dim].is_monotonic_increasing:
                raise RuntimeError(
                    "Lead times remain non-monotonic for "
                    f"initialization time {time}: "
                    f"{combined_time[lead_time_dim].values}"
                )

            data_time = _sort_sample_coords(combined_time)

            if post_processor is not None:
                data_time = post_processor.to_dataset(data_time)
                data_time = post_processor.inverse_transform(data_time)
            else:
                data_time = data_time.to_dataset(dim="channels")

            year_datasets.append(data_time)

        if not year_datasets:
            continue

        data_year = xr.concat(
            year_datasets,
            dim=init_time_dim,
            coords="minimal",
            compat="equals",
            join="exact",
        )

        data_year = _sort_sample_coords(data_year)

        if not data_year.indexes[init_time_dim].is_monotonic_increasing:
            raise RuntimeError(
                "Initialization times remain non-monotonic for "
                f"year {year}: "
                f"{data_year[init_time_dim].values}"
            )

        save_name_ = f"{source}_{int(year)}"

        if nstds is not None:
            save_name_ += f"_nstds{nstds}"
        if output_ensemble:
            save_name_ += "_output_ensemble"

        save_name = save_name_ + ".nc"

        output_path = output_dir / save_name

        if post_processor is not None:
            data_year = post_processor.inverse_rename(data_year)

        data_year.to_netcdf(output_path)
        data_year.close()

        for data in year_datasets:
            data.close()

        if logger_function is not None:
            logger_function(
                logging.INFO,
                "Saved aggregated predictions for "
                f"initialization year {year}: {output_path}",
            )

    if cleanup_temp:
        for path in temp_files:
            path.unlink()

        temp_save_dir.rmdir()
