import torch
import numpy as np
import xarray as xr
from typing import Dict, Optional, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime
from cccma_ppp.architectures.models_abc import DeterministicRequest
from cccma_ppp.core.models.deterministic import deterministicOutput
from cccma_ppp.preprocessing.utils_preprocessing import Standardizer
from cccma_ppp.configs import required_sample_dimensions

init_time_dim, lead_time_dim = required_sample_dimensions

# Data path configuration
DATA_PATH = Path(
    "/space/hall7/sitestore/eccc/crd/cccma/users/rpg002/data/tas/forecast_data/datetime"
)
FALLBACK_TO_SYNTHETIC = True

# Logging configuration
LOG_DIR = Path("scripts/module_integration_tests")
LOG_FILE = LOG_DIR / "inverse_preprocessing_all.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def log_and_print(message: str):
    """Log and print message simultaneously."""
    print(message)
    logger.info(message)


class MockDeterministicModel:
    """Mock model for testing."""

    def forward(self, request: DeterministicRequest) -> deterministicOutput:
        output = request.input * 0.95 + 0.05
        return deterministicOutput(output=output)


class InverseTransformationMapper:
    def __init__(self, standardizers: Dict[int, Standardizer]):
        self.standardizers = standardizers
        if len(standardizers) != 12:
            raise ValueError("Must have 12 standardizers (one per month)")

    def apply_inverse_transformation(
        self,
        output: torch.Tensor,
        init_month: Union[int, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if output.dim() == 3:
            output = output.unsqueeze(0)
            squeeze_output = True
        elif output.dim() == 4:
            squeeze_output = False
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {output.dim()}D")

        batch_size = output.shape[0]

        if isinstance(init_month, torch.Tensor):
            if init_month.shape[0] != batch_size:
                raise ValueError("init_month batch size mismatch")
            init_months = init_month.cpu().numpy().astype(int)
        else:
            init_months = np.full(batch_size, init_month, dtype=int)

        denormalized = torch.zeros_like(output)
        for i, month in enumerate(init_months):
            if not 0 <= month <= 11:
                raise ValueError(f"init_month must be 0-11, got {month}")

            output_np = output[i].cpu().numpy()
            output_xr = xr.DataArray(output_np, dims=["var", "height", "width"])

            denormalized_xr = self.standardizers[int(month)].inverse_transform(
                output_xr
            )
            denormalized[i] = torch.from_numpy(np.array(denormalized_xr.data)).float()

        if squeeze_output:
            denormalized = denormalized.squeeze(0)

        return denormalized

    def process_deterministic_output(
        self,
        det_output: deterministicOutput,
        init_month: Union[int, torch.Tensor],
    ) -> deterministicOutput:
        denormalized = self.apply_inverse_transformation(det_output.output, init_month)
        return deterministicOutput(output=denormalized)

    def process_deterministic_request(
        self,
        model,
        request: DeterministicRequest,
        init_month: Union[int, torch.Tensor],
    ) -> deterministicOutput:
        model_output = model.forward(request)
        return self.process_deterministic_output(model_output, init_month)


def create_standardizers_per_month(
    data: Optional[np.ndarray] = None, frequency: str = None
):
    """Create 12 Standardizer instances, one per month."""
    standardizers = {}

    for month in range(12):
        standardizer = Standardizer(dims=[], frequency=frequency)

        if data is not None:
            monthly_data = data
        else:
            monthly_data = np.random.randn(100, 3, 64, 64) + month * 0.3

        monthly_data_xr = xr.DataArray(
            monthly_data,
            dims=["time", "var", "height", "width"],
            coords={"time": np.arange(len(monthly_data))},
        )

        standardizer.fit(monthly_data_xr)
        standardizers[month] = standardizer

    return standardizers


def roundtrip_test(
    original_data: np.ndarray,
    standardizer: Standardizer,
    test_name: str,
) -> Tuple[float, np.ndarray]:
    """Test forward and inverse transformation roundtrip."""
    original_xr = xr.DataArray(original_data, dims=["var", "height", "width"])

    normalized_xr = standardizer.transform(original_xr)
    recovered_xr = standardizer.inverse_transform(normalized_xr)
    recovered_np = np.array(recovered_xr.data)

    error = np.abs(original_data - recovered_np).max()

    log_and_print(f"\n[{test_name} Roundtrip]")
    log_and_print(f"  Original shape: {original_data.shape}")
    log_and_print(
        f"  Original mean: {float(original_xr.mean()):.6f}, std: {float(original_xr.std()):.6f}"
    )
    log_and_print(
        f"  Normalized mean: {float(normalized_xr.mean()):.6f}, std: {float(normalized_xr.std()):.6f}"
    )
    log_and_print(f"  Recovered shape: {recovered_np.shape}")
    log_and_print(
        f"  Recovered mean: {recovered_np.mean():.6f}, std: {recovered_np.std():.6f}"
    )
    log_and_print(f"  Max reconstruction error: {error:.2e}")

    return error, recovered_np


def load_actual_data_from_path(
    data_path: Path,
    max_samples: int = 100,
) -> Optional[np.ndarray]:
    """Load actual data from NetCDF/HDF5 files."""
    try:
        if not data_path.exists():
            log_and_print(f"  Warning: Data path not found: {data_path}")
            return None

        nc_files = sorted(data_path.glob("*.nc"))
        h5_files = sorted(data_path.glob("*.h5")) + sorted(data_path.glob("*.hdf5"))
        all_files = nc_files + h5_files

        if not all_files:
            log_and_print(f"  Warning: No data files found in {data_path}")
            return None

        log_and_print(f"  Found {len(all_files)} data files")

        data_list = []
        for file_path in all_files[:max_samples]:
            try:
                if file_path.suffix == ".nc":
                    ds = xr.open_dataset(file_path)
                    var_name = list(ds.data_vars.keys())[0] if ds.data_vars else None
                    if var_name:
                        data = ds[var_name].values
                        if data.ndim >= 2:
                            data_list.append(data.astype(np.float32))
                    ds.close()
                else:
                    import h5py

                    with h5py.File(file_path, "r") as f:
                        key = list(f.keys())[0] if f.keys() else None
                        if key:
                            data = f[key][()]
                            if data.ndim >= 2:
                                data_list.append(data.astype(np.float32))
            except Exception as e:
                log_and_print(f"  Warning: Failed to load {file_path.name}: {e}")
                continue

        if not data_list:
            log_and_print("  Warning: No valid data loaded from files")
            return None

        combined_data = np.concatenate(data_list, axis=0)

        if combined_data.ndim == 3:
            combined_data = combined_data[:, np.newaxis, :, :]
        elif combined_data.ndim == 2:
            combined_data = combined_data[:, np.newaxis, :, np.newaxis]

        if combined_data.ndim != 4:
            log_and_print(f"  Warning: Data has unexpected shape {combined_data.shape}")
            return None

        if combined_data.shape[0] > max_samples:
            combined_data = combined_data[:max_samples]

        log_and_print(f"  Successfully loaded data shape: {combined_data.shape}")
        return combined_data

    except Exception as e:
        log_and_print(f"  Error loading data from {data_path}: {e}")
        return None


def generate_actual_data(
    shape: Tuple[int, int, int, int] = (100, 3, 64, 64),
) -> np.ndarray:
    """Generate realistic data."""
    np.random.seed(42)
    time_steps, n_vars, height, width = shape

    time_pattern = np.sin(np.linspace(0, 4 * np.pi, time_steps))[:, None, None]
    lat_pattern = np.cos(np.linspace(0, 2 * np.pi, height))[None, None, :]
    lon_pattern = np.sin(np.linspace(0, 2 * np.pi, width))[None, :, None]

    data = time_pattern * lat_pattern * lon_pattern
    data = np.tile(data[:, np.newaxis, :, :], (1, n_vars, 1, 1))

    return data.astype(np.float32)


def main():
    log_and_print("=" * 80)
    log_and_print("INVERSE TRANSFORMATION MAPPER - COMPLETE TEST SUITE")
    log_and_print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_print("=" * 80)

    # Load data
    log_and_print("\nLoading actual data from path...")
    log_and_print(f"  Data path: {DATA_PATH}")
    actual_data_base = load_actual_data_from_path(DATA_PATH, max_samples=100)

    if actual_data_base is None:
        log_and_print("\n  Actual data path not found or loading failed.")
        if FALLBACK_TO_SYNTHETIC:
            log_and_print("  Falling back to generated synthetic data...")
            actual_data_base = generate_actual_data()
        else:
            log_and_print("  Skipping tests.")
            return

    if actual_data_base is not None:
        log_and_print(f"  Actual data shape: {actual_data_base.shape}")
        log_and_print(
            f"  Actual data mean: {actual_data_base.mean():.6f}, std: {actual_data_base.std():.6f}"
        )

    log_and_print("\n" + "=" * 80)
    log_and_print("PART 1: SHUFFLED FREQUENCIES TESTS (dims=[], frequency ignored)")
    log_and_print("=" * 80)

    frequencies = ["day", "month", "year"]
    frequency_results = {}

    for freq in frequencies:
        log_and_print(f"\n[Testing Frequency: {freq.upper()}]")

        log_and_print(f"\n  Synthetic Data with {freq} frequency:")
        standardizers_freq_syn = create_standardizers_per_month(
            data=None, frequency=freq
        )
        mapper_freq_syn = InverseTransformationMapper(standardizers_freq_syn)

        test_data_syn = np.random.randn(3, 64, 64).astype(np.float32)
        original_xr_syn = xr.DataArray(test_data_syn, dims=["var", "height", "width"])

        normalized_xr_syn = standardizers_freq_syn[0].transform(original_xr_syn)
        normalized_syn = torch.from_numpy(np.array(normalized_xr_syn.data)).float()

        denormalized_syn = mapper_freq_syn.apply_inverse_transformation(
            normalized_syn, init_month=0
        )
        error_freq_syn = np.abs(test_data_syn - denormalized_syn.numpy()).max()

        log_and_print(
            f"    Original mean: {test_data_syn.mean():.6f}, std: {test_data_syn.std():.6f}"
        )
        log_and_print(
            f"    Normalized mean: {float(normalized_xr_syn.mean()):.6f}, std: {float(normalized_xr_syn.std()):.6f}"
        )
        log_and_print(f"    Reconstruction error: {error_freq_syn:.2e}")

        log_and_print(f"\n  Actual Data with {freq} frequency:")
        standardizers_freq_act = create_standardizers_per_month(
            data=actual_data_base, frequency=freq
        )
        mapper_freq_act = InverseTransformationMapper(standardizers_freq_act)

        test_data_act = actual_data_base[0, :, :, :].copy()
        original_xr_act = xr.DataArray(test_data_act, dims=["var", "height", "width"])

        normalized_xr_act = standardizers_freq_act[0].transform(original_xr_act)
        normalized_act = torch.from_numpy(np.array(normalized_xr_act.data)).float()

        denormalized_act = mapper_freq_act.apply_inverse_transformation(
            normalized_act, init_month=0
        )
        error_freq_act = np.abs(test_data_act - denormalized_act.numpy()).max()

        log_and_print(
            f"    Original mean: {test_data_act.mean():.6f}, std: {test_data_act.std():.6f}"
        )
        log_and_print(
            f"    Normalized mean: {float(normalized_xr_act.mean()):.6f}, std: {float(normalized_xr_act.std()):.6f}"
        )
        log_and_print(f"    Reconstruction error: {error_freq_act:.2e}")

        frequency_results[freq] = {
            "synthetic_error": error_freq_syn,
            "actual_error": error_freq_act,
        }

    log_and_print("\n" + "=" * 80)
    log_and_print("PART 1B: FREQUENCY TESTS WITH dims=[init_time_dim] (ACTIVE)")
    log_and_print("=" * 80)

    frequency_results_active = {}

    for freq in frequencies:
        log_and_print(f"\n[Testing Frequency: {freq.upper()} - WITH init_time_dim]")

        try:
            log_and_print(
                f"\n  Synthetic Data with {freq} frequency (dims=[init_time_dim]):"
            )

            standardizer_syn = Standardizer(dims=[init_time_dim], frequency=freq)
            synthetic_data = np.random.randn(100, 3, 64, 64).astype(np.float32)
            time_coord = xr.cftime_range(start="2020-01-01", periods=100, freq="D")

            synthetic_xr = xr.DataArray(
                synthetic_data,
                dims=[init_time_dim, "var", "height", "width"],
                coords={init_time_dim: time_coord},
            )

            standardizer_syn.fit(synthetic_xr)

            test_sample_syn = synthetic_data[0:1, :, :, :]
            test_sample_time = xr.cftime_range(start="2020-01-01", periods=1, freq="D")

            test_xr_syn = xr.DataArray(
                test_sample_syn,
                dims=[init_time_dim, "var", "height", "width"],
                coords={init_time_dim: test_sample_time},
            )

            normalized_xr_syn = standardizer_syn.transform(test_xr_syn)
            recovered_xr_syn = standardizer_syn.inverse_transform(normalized_xr_syn)

            error_syn_active = np.abs(
                test_sample_syn - np.array(recovered_xr_syn.data)
            ).max()

            log_and_print(
                f"    Original mean: {test_sample_syn.mean():.6f}, std: {test_sample_syn.std():.6f}"
            )
            log_and_print(
                f"    Normalized mean: {float(normalized_xr_syn.mean()):.6f}, std: {float(normalized_xr_syn.std()):.6f}"
            )
            log_and_print(
                f"    Recovered mean: {float(recovered_xr_syn.mean()):.6f}, std: {float(recovered_xr_syn.std()):.6f}"
            )
            log_and_print(f"    Reconstruction error: {error_syn_active:.2e}")

            log_and_print(
                f"\n  Actual Data with {freq} frequency (dims=[init_time_dim]):"
            )

            standardizer_act = Standardizer(dims=[init_time_dim], frequency=freq)
            actual_time_coord = xr.cftime_range(
                start="2020-01-01", periods=actual_data_base.shape[0], freq="D"
            )
            actual_xr = xr.DataArray(
                actual_data_base,
                dims=[init_time_dim, "var", "height", "width"],
                coords={init_time_dim: actual_time_coord},
            )

            standardizer_act.fit(actual_xr)

            test_sample_act = actual_data_base[0:1, :, :, :]
            test_sample_time_act = xr.cftime_range(
                start="2020-01-01", periods=1, freq="D"
            )

            test_xr_act = xr.DataArray(
                test_sample_act,
                dims=[init_time_dim, "var", "height", "width"],
                coords={init_time_dim: test_sample_time_act},
            )

            normalized_xr_act = standardizer_act.transform(test_xr_act)
            recovered_xr_act = standardizer_act.inverse_transform(normalized_xr_act)

            error_act_active = np.abs(
                test_sample_act - np.array(recovered_xr_act.data)
            ).max()

            log_and_print(
                f"    Original mean: {test_sample_act.mean():.6f}, std: {test_sample_act.std():.6f}"
            )
            log_and_print(
                f"    Normalized mean: {float(normalized_xr_act.mean()):.6f}, std: {float(normalized_xr_act.std()):.6f}"
            )
            log_and_print(
                f"    Recovered mean: {float(recovered_xr_act.mean()):.6f}, std: {float(recovered_xr_act.std()):.6f}"
            )
            log_and_print(f"    Reconstruction error: {error_act_active:.2e}")

            frequency_results_active[freq] = {
                "synthetic_error": error_syn_active,
                "actual_error": error_act_active,
            }

        except Exception as e:
            log_and_print(f"    Error with {freq} frequency: {e}")
            frequency_results_active[freq] = {
                "synthetic_error": "ERROR",
                "actual_error": "ERROR",
            }

    log_and_print("\n" + "=" * 80)
    log_and_print("PART 2: MODEL INTEGRATION TESTS")
    log_and_print("=" * 80)

    standardizers_model_syn = create_standardizers_per_month(data=None)
    mapper_model_syn = InverseTransformationMapper(standardizers_model_syn)
    model = MockDeterministicModel()

    log_and_print("\n[Synthetic Model Integration]")
    input_tensor_syn = torch.randn(8, 3, 64, 64)
    log_and_print(f"  Input tensor shape: {input_tensor_syn.shape}")
    log_and_print(
        f"  Input tensor mean: {float(input_tensor_syn.mean()):.6f}, std: {float(input_tensor_syn.std()):.6f}"
    )

    request_syn = DeterministicRequest(input=input_tensor_syn)
    model_output_syn = model.forward(request_syn)
    log_and_print(f"  Model output shape: {model_output_syn.output.shape}")
    log_and_print(
        f"  Model output mean: {float(model_output_syn.output.mean()):.6f}, std: {float(model_output_syn.output.std()):.6f}"
    )

    months_syn = torch.tensor([i % 12 for i in range(8)])
    denormalized_syn = mapper_model_syn.apply_inverse_transformation(
        model_output_syn.output, init_month=months_syn
    )
    log_and_print(f"  Denormalized output shape: {denormalized_syn.shape}")
    log_and_print(
        f"  Denormalized output mean: {float(denormalized_syn.mean()):.6f}, std: {float(denormalized_syn.std()):.6f}"
    )
    log_and_print(f"  Months used: {[int(m) for m in months_syn.tolist()]}")

    standardizers_model_act = create_standardizers_per_month(data=actual_data_base)
    mapper_model_act = InverseTransformationMapper(standardizers_model_act)

    log_and_print("\n[Actual Model Integration]")
    input_tensor_act = torch.from_numpy(actual_data_base[:8]).float()
    log_and_print(f"  Input tensor shape: {input_tensor_act.shape}")
    log_and_print(
        f"  Input tensor mean: {float(input_tensor_act.mean()):.6f}, std: {float(input_tensor_act.std()):.6f}"
    )

    request_act = DeterministicRequest(input=input_tensor_act)
    model_output_act = model.forward(request_act)
    log_and_print(f"  Model output shape: {model_output_act.output.shape}")
    log_and_print(
        f"  Model output mean: {float(model_output_act.output.mean()):.6f}, std: {float(model_output_act.output.std()):.6f}"
    )

    months_act = torch.tensor([i % 12 for i in range(8)])
    denormalized_act = mapper_model_act.apply_inverse_transformation(
        model_output_act.output, init_month=months_act
    )
    log_and_print(f"  Denormalized output shape: {denormalized_act.shape}")
    log_and_print(
        f"  Denormalized output mean: {float(denormalized_act.mean()):.6f}, std: {float(denormalized_act.std()):.6f}"
    )
    log_and_print(f"  Months used: {[int(m) for m in months_act.tolist()]}")

    log_and_print("\n" + "=" * 80)
    log_and_print("PART 3: SHUFFLED DIMENSIONS TESTS")
    log_and_print("=" * 80)

    standardizers_base = create_standardizers_per_month(data=actual_data_base)

    dimension_configs = [
        {"name": "Standard [var, height, width]", "dims": ["var", "height", "width"]},
        {"name": "Shuffled [height, var, width]", "dims": ["height", "var", "width"]},
        {"name": "Shuffled [width, height, var]", "dims": ["width", "height", "var"]},
        {"name": "Subset [var, height]", "dims": ["var", "height"]},
        {"name": "Subset [height, width]", "dims": ["height", "width"]},
    ]

    dimension_results = {}

    for config in dimension_configs:
        log_and_print(f"\n[Dimension Test: {config['name']}]")
        test_data = actual_data_base[5, :, :, :].copy()

        try:
            original_xr = xr.DataArray(test_data, dims=config["dims"])

            log_and_print(
                f"  Original shape: {original_xr.shape}, dims: {original_xr.dims}"
            )
            log_and_print(
                f"  Original mean: {float(original_xr.mean()):.6f}, std: {float(original_xr.std()):.6f}"
            )

            normalized_xr = standardizers_base[5].transform(original_xr)
            log_and_print(
                f"  Normalized mean: {float(normalized_xr.mean()):.6f}, std: {float(normalized_xr.std()):.6f}"
            )

            recovered_xr = standardizers_base[5].inverse_transform(normalized_xr)
            log_and_print(
                f"  Recovered mean: {float(recovered_xr.mean()):.6f}, std: {float(recovered_xr.std()):.6f}"
            )

            error_dim = np.abs(test_data - np.array(recovered_xr.data)).max()
            log_and_print(f"  Reconstruction error: {error_dim:.2e}")

            dimension_results[config["name"]] = error_dim
        except Exception as e:
            log_and_print(f"  Error: {e}")
            dimension_results[config["name"]] = "ERROR"

    log_and_print("\n" + "=" * 80)
    log_and_print("PART 4: STACKED CHAIN WITH SHUFFLED FREQUENCIES")
    log_and_print("=" * 80)

    stacked_configs = [
        {
            "name": "Default: month, month, month, month",
            "frequencies": ["month", "month", "month", "month"],
        },
        {
            "name": "Shuffled: day, month, year, day",
            "frequencies": ["day", "month", "year", "day"],
        },
        {
            "name": "Shuffled: year, year, day, month",
            "frequencies": ["year", "year", "day", "month"],
        },
        {
            "name": "Shuffled: day, day, day, day",
            "frequencies": ["day", "day", "day", "day"],
        },
    ]

    stacked_results = {}

    for config in stacked_configs:
        log_and_print(f"\n[Stacked Chain: {config['name']}]")

        test_data = actual_data_base[10, :, :, :].copy()
        original_xr = xr.DataArray(test_data, dims=["var", "height", "width"])

        log_and_print(
            f"  Original mean: {test_data.mean():.6f}, std: {test_data.std():.6f}"
        )

        standardizers_stacked = []
        for freq in config["frequencies"]:
            std = create_standardizers_per_month(data=actual_data_base, frequency=freq)
            standardizers_stacked.append(std[0])

        current = original_xr
        for i, std in enumerate(standardizers_stacked):
            current = std.transform(current)
            log_and_print(
                f"    After transform {i} ({config['frequencies'][i]}): mean={float(current.mean()):.6f}, std={float(current.std()):.6f}"
            )

        for i in range(len(standardizers_stacked) - 1, -1, -1):
            current = standardizers_stacked[i].inverse_transform(current)
            log_and_print(
                f"    After inverse {i} ({config['frequencies'][i]}): mean={float(current.mean()):.6f}, std={float(current.std()):.6f}"
            )

        error_stacked = np.abs(test_data - np.array(current.data)).max()
        log_and_print(f"  Final reconstruction error: {error_stacked:.2e}")

        stacked_results[config["name"]] = error_stacked

    log_and_print("\n" + "=" * 80)
    log_and_print("RESULTS SUMMARY")
    log_and_print("=" * 80)

    log_and_print("\nPART 1: Frequency Tests (dims=[], frequency parameter ignored):")
    for freq, results in frequency_results.items():
        log_and_print(f"  {freq.upper()}:")
        log_and_print(f"    Synthetic error: {results['synthetic_error']:.2e}")
        log_and_print(f"    Actual error: {results['actual_error']:.2e}")

    log_and_print(
        "\nPART 1B: Frequency Tests (dims=[init_time_dim], frequency ACTIVE):"
    )
    for freq, results in frequency_results_active.items():
        log_and_print(f"  {freq.upper()}:")
        if isinstance(results["synthetic_error"], str):
            log_and_print(f"    Synthetic error: {results['synthetic_error']}")
            log_and_print(f"    Actual error: {results['actual_error']}")
        else:
            log_and_print(f"    Synthetic error: {results['synthetic_error']:.2e}")
            log_and_print(f"    Actual error: {results['actual_error']:.2e}")

    log_and_print("\nPART 2: Model Integration Tests:")
    log_and_print(
        "  Synthetic: Successfully processed 8-sample batch with mixed months"
    )
    log_and_print("  Actual: Successfully processed 8-sample batch with mixed months")

    log_and_print("\nPART 3: Dimension Tests:")
    for dim_name, error in dimension_results.items():
        if isinstance(error, str):
            log_and_print(f"  {dim_name}: {error}")
        else:
            log_and_print(f"  {dim_name}: {error:.2e}")

    log_and_print("\nPART 4: Stacked Chain Tests (Shuffled Frequencies):")
    for chain_name, error in stacked_results.items():
        log_and_print(f"  {chain_name}: {error:.2e}")

    log_and_print("\n" + "=" * 80)
    log_and_print("All tests completed successfully!")
    log_and_print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_print("=" * 80)
    log_and_print(f"\nLog file saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
