import torch
from pathlib import Path
import logging
import dataclasses
import os
import time
from tqdm import tqdm
import numpy as np
import xarray as xr

from S2D4P.core.trainer import clear_memory

from S2D4P.data.dataloader import Dataloader
from S2D4P.generic.distributed import Distributed
from S2D4P.generic.runtime import RuntimeContext
from S2D4P.generic.aggregator import RunningCovariance
from S2D4P.train.dataloader import TrainDataloaderConfig
from S2D4P.core.core_abc import moduleABC

from S2D4P.preprocessing.preprocessing import PreprocessingPipeline

from S2D4P.inference.predictors.deterministic import DeterministicPredictorConfig
from S2D4P.inference.predictors.cvae import cVAEPredictorConfig

from S2D4P.configs import required_sample_dimensions, realization_dim

init_time_dim, lead_time_dim = required_sample_dimensions


@dataclasses.dataclass
class WriterConfig:
    """
    Configuration for inference prediction and output writing.

    Parameters
    ----------
    predictor : DeterministicPredictorConfig | cVAEPredictorConfig | None
        Predictor configuration used to generate inference outputs. If ``None``,
        a deterministic predictor is used by default when prediction is enabled.
    num_output_sampling : int
        Number of output samples generated for each prediction.
    get_trained_model_stats_from_validation : bool
        Whether model statistics required for inference are estimated from the
        validation dataset instead of the training dataset.
    aggregate_only : bool
        Whether to skip prediction and aggregate previously saved temporary
        prediction files only.
    predict_only : bool
        Whether to generate temporary prediction files without aggregating them
        into final NetCDF outputs.
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
        Build an inference writer.

        The predictor type is checked against the loaded model type when prediction
        is enabled.

        Parameters
        ----------
        inference_data_loader : Dataloader
            Data loader providing batches used for inference.
        train_dataloader_config : TrainDataloaderConfig
            Training data-loader configuration used when training statistics must be
            reconstructed.
        module : moduleABC
            Trained model module used for prediction.
        post_processor : PreprocessingPipeline
            Fitted preprocessing pipeline used to reconstruct and inverse transform
            model outputs.
        output_dir : Path | str
            Directory in which temporary and aggregated outputs are stored.

        Returns
        -------
        Writer
            Configured inference writer.

        Raises
        ------
        RuntimeError
            If both aggregation-only and prediction-only modes are enabled, or the
            configured predictor type does not match the model module type.
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
    Run inference and aggregate prediction outputs.

    The writer coordinates distributed prediction, extraction of training-derived
    model statistics, temporary batch output storage, postprocessing, and
    year-by-year NetCDF aggregation.

    Parameters
    ----------
    config : WriterConfig
        Configuration controlling prediction and aggregation behavior.
    inference_data_loader : Dataloader
        Data loader providing inference batches.
    train_dataloader_config : TrainDataloaderConfig
        Training data-loader configuration used to reconstruct training or
        validation loaders when statistics are required.
    module : moduleABC
        Trained model module used for inference.
    post_processor : PreprocessingPipeline
        Fitted preprocessing pipeline used to reconstruct and inverse transform
        predictions.
    output_dir : Path | str
        Directory in which inference outputs are stored.
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
            Data loader providing inference batches.
        train_dataloader_config : TrainDataloaderConfig
            Training data-loader configuration used for model-statistic extraction.
        module : moduleABC
            Trained model module used for inference.
        post_processor : PreprocessingPipeline
            Fitted preprocessing pipeline used for output reconstruction.
        output_dir : Path | str
            Directory in which inference outputs are stored.
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
        Configure the writer for distributed inference.

        Establishes distributed process state, validates model device placement,
        prepares temporary output storage, builds the configured predictor, and
        extracts required training statistics when necessary.

        Parameters
        ----------
        distributed : Distributed
            Distributed runtime containing device, rank, local-rank, and world-size
            information.
        logger : logging.Logger
            Logger used for inference status messages.

        Raises
        ------
        RuntimeError
            If the model is not on the expected distributed device, or aggregation
            mode is requested without existing temporary NetCDF prediction files.
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
        Return the path used to store training-derived model statistics.

        Returns
        -------
        Path
            Path to ``training_variable_stats.pt`` under the output directory.
        """
        return Path(self.output_dir) / "training_variable_stats.pt"

    def predict(self):
        """
        Run prediction and output aggregation.

        Inference timing is recorded and the configured prediction or aggregation
        workflow is executed.

        Raises
        ------
        RuntimeError
            If distributed writer setup has not been completed.
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
        Run prediction and output aggregation.

        Inference timing is recorded and the configured prediction or aggregation
        workflow is executed.

        Raises
        ------
        RuntimeError
            If distributed writer setup has not been completed.
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
        Calculate and save training-derived model statistics.

        Statistics are collected only when the saved statistics file does not already
        exist. Each required running covariance is accumulated over the selected
        training or validation loader and reduced across distributed processes before
        being saved.
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
        Build a training or validation loader for statistic extraction.

        The training data-loader configuration is initialized using the preprocessing
        pipelines saved with the experiment.

        Parameters
        ----------
        from_validation : bool
            Whether to build the validation loader instead of the training loader.
        return_metadata : bool
            Whether samples include their sampling-coordinate metadata.
        shuffle : bool | None
            Whether samples are shuffled.

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
        Aggregate and save training-derived covariance statistics.

        Running statistics are reduced across distributed processes. The root process
        then calculates each mean and covariance matrix and saves the results.

        Parameters
        ----------
        stats : dict[str, RunningCovariance]
            Running covariance accumulators keyed by statistic name.
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
        Log a message from the root process.

        Parameters
        ----------
        level : int
            Logging level used for the message.
        msg : str
            Message to log or print.
        args : Any
            Positional arguments passed to the configured logger.
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
            Module wrapped by distributed data parallel execution when applicable, or
            the original module otherwise.
        """

        if isinstance(self.module, torch.nn.parallel.DistributedDataParallel):
            return self.module.module
        return self.module

    def aggregate_predictions_to_netcdf(self, do_post_process: bool = True):
        """
        Aggregate temporary prediction files into final NetCDF outputs.

        Aggregation is performed only by the root process after distributed ranks are
        synchronized. The fitted output preprocessing pipeline can optionally be
        disabled.

        Parameters
        ----------
        do_post_process : bool
            Whether predictions are reconstructed and inverse transformed with the
            fitted output preprocessing pipeline.
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

    Temporary files from all ranks are scanned to determine available
    initialization times. Predictions are combined first by initialization time
    and lead time, then by initialization year. Coordinates are sorted,
    duplicates are removed from lead times, and optional preprocessing is
    inverted before the yearly datasets are saved.

    Parameters
    ----------
    post_processor : PreprocessingPipeline | None
        Fitted preprocessing pipeline used to reconstruct and inverse transform
        predictions. If ``None``, prediction channels are converted directly to
        dataset variables.
    output_dir : Path
        Directory containing the ``_temp`` prediction directory and receiving the
        final yearly NetCDF files.
    output_ensemble : bool
        Whether output filenames indicate that predictions contain an output
        ensemble.
    logger_function : callable | None
        Optional logging function receiving a logging level and message.
    cleanup_temp : bool
        Whether temporary prediction files and their directory are removed after
        successful aggregation.
    init_time_dim : str
        Name of the initialization-time dimension.
    lead_time_dim : str
        Name of the lead-time dimension.
    optional_sample_dimensions : tuple[str, ...]
        Additional sample dimensions sorted and cleaned during aggregation.

    Raises
    ------
    RuntimeError
        If no temporary prediction files are available, required initialization
        coordinates are missing, no initialization times are present, or lead or
        initialization times remain non-monotonic after sorting.
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
        Sort available sample coordinates in a prediction object.

        Parameters
        ----------
        data : xr.DataArray | xr.Dataset
            Prediction data whose sample coordinates are sorted.

        Returns
        -------
        xr.DataArray | xr.Dataset
            Prediction data sorted by each available configured sample coordinate.
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
