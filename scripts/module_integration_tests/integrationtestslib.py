from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")
logging.raiseExceptions = False


class StringUtils:
    """String manipulation utilities"""

    @staticmethod
    def clean_name(value: Any) -> str:
        """Convert value to filesystem-safe name"""
        value = str(value).replace("None", "none")
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
        value = re.sub(r"_+", "_", value)
        return value.strip("_")

    @staticmethod
    def serialize_json(value: Any) -> str:
        """Convert value to JSON string"""
        return json.dumps(value, sort_keys=True, default=str)


class LoggingUtils:
    """Logging utilities"""

    @staticmethod
    def cleanup_handlers() -> None:
        """Close all logging handlers"""
        logger_names = ["", "training", "inference"]
        logger_names.extend(
            name
            for name in logging.Logger.manager.loggerDict
            if name.startswith("cccma_ppp")
        )

        for name in dict.fromkeys(logger_names):
            logger = logging.getLogger(name)
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                try:
                    handler.flush()
                except Exception:
                    pass
                try:
                    handler.close()
                except Exception:
                    pass

    @staticmethod
    def append_exception(log_path: Path) -> None:
        """Append exception traceback to log file"""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            traceback.print_exc(file=file)

    @staticmethod
    def run_silenced(function: Callable[[], Any], log_path: Path) -> Any:
        """Run function and capture output to log file"""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        LoggingUtils.cleanup_handlers()

        with log_path.open("w", encoding="utf-8") as log_file:
            try:
                with redirect_stdout(log_file), redirect_stderr(log_file):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        return function()
            finally:
                LoggingUtils.cleanup_handlers()


class TensorUtils:
    """Tensor manipulation utilities"""

    @staticmethod
    def scalar_value(value: Any) -> Any:
        """Convert tensors/arrays to Python scalars"""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def canonical_value(value: Any) -> Any:
        """Convert values to canonical form for consistent JSON serialization"""
        value = TensorUtils.scalar_value(value)

        if isinstance(value, dict):
            return {
                str(key): TensorUtils.canonical_value(child)
                for key, child in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [TensorUtils.canonical_value(child) for child in value]
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, np.datetime64):
            return str(value.astype("datetime64[ns]"))
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass

        return value

    @staticmethod
    def extract_metadata(batch: Any) -> Any:
        """Extract metadata from batch"""
        if isinstance(batch, dict):
            return batch.get("metadata")
        return getattr(batch, "metadata", None)

    @staticmethod
    def flatten_metadata(metadata: Any) -> list:
        """Convert metadata to list of dicts"""
        if metadata is None:
            return []
        if isinstance(metadata, list):
            return metadata
        if isinstance(metadata, tuple):
            return list(metadata)
        if isinstance(metadata, dict):
            lengths = [
                len(value)
                for value in metadata.values()
                if isinstance(value, (list, tuple, torch.Tensor, np.ndarray))
            ]
            if not lengths:
                return [metadata]

            count = lengths[0]
            return [
                {
                    key: TensorUtils.scalar_value(value[i])
                    if isinstance(value, (list, tuple))
                    else TensorUtils.scalar_value(value)
                    for key, value in metadata.items()
                }
                for i in range(count)
            ]
        return [metadata]

    @staticmethod
    def analyze_mask(mask: torch.Tensor, name: str = "mask") -> dict[str, Any]:
        """Analyze properties of a spatial mask tensor"""
        if mask is None:
            return {"present": False}

        if not isinstance(mask, torch.Tensor):
            return {
                "present": False,
                "error": f"Expected tensor, got {type(mask).__name__}",
            }

        mask_np = mask.detach().cpu().numpy()
        is_binary = bool(np.all(np.isin(mask_np, [0, 1])))

        return {
            "present": True,
            "shape": tuple(mask.shape),
            "dtype": str(mask.dtype),
            "is_binary": is_binary,
            "valid_count": int(np.sum(mask_np == 1)) if is_binary else None,
            "valid_percentage": round(100.0 * np.sum(mask_np == 1) / mask_np.size, 2)
            if is_binary
            else None,
        }


class PreprocessingUtils:
    """Preprocessing-specific utilities for fitted pipeline inspection"""

    @staticmethod
    def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
        """Return tensor statistics"""
        cpu_value = value.detach().cpu()
        result: dict[str, Any] = {
            "shape": list(cpu_value.shape),
            "dtype": str(cpu_value.dtype),
            "device": str(value.device),
            "numel": int(cpu_value.numel()),
        }

        if torch.is_floating_point(cpu_value):
            finite_values = cpu_value[torch.isfinite(cpu_value)]
            result.update(
                {
                    "finite_count": int(finite_values.numel()),
                    "nan_count": int(torch.isnan(cpu_value).sum().item()),
                    "infinite_count": int(torch.isinf(cpu_value).sum().item()),
                    "minimum": float(finite_values.min().item())
                    if finite_values.numel()
                    else None,
                    "maximum": float(finite_values.max().item())
                    if finite_values.numel()
                    else None,
                    "mean": float(finite_values.float().mean().item())
                    if finite_values.numel()
                    else None,
                    "standard_deviation": float(finite_values.float().std().item())
                    if finite_values.numel() > 1
                    else 0.0,
                }
            )

        return result

    @staticmethod
    def array_summary(array: Any) -> dict[str, Any]:
        """Create summary for NumPy array or xarray"""
        try:
            import xarray as xr
        except ImportError:
            xr = None

        # Handle xarray
        if xr is not None and isinstance(array, xr.DataArray):
            arr_np = np.asarray(array.values)
            result = {
                "type": "xarray.DataArray",
                "name": str(array.name),
                "dims": list(array.dims),
                "coords": list(array.coords.keys()),
            }
            result.update(PreprocessingUtils._numeric_summary(arr_np))
            return result

        if xr is not None and isinstance(array, xr.Dataset):
            return {
                "type": "xarray.Dataset",
                "dimensions": {
                    str(name): int(size) for name, size in array.sizes.items()
                },
                "variables": list(array.data_vars),
                "coords": list(array.coords.keys()),
            }

        # Handle NumPy
        if isinstance(array, np.ndarray):
            result = {
                "type": "numpy.ndarray",
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "size": int(array.size),
            }
            result.update(PreprocessingUtils._numeric_summary(array))
            return result

        return {"type": type(array).__name__, "representation": repr(array)[:100]}

    @staticmethod
    def _numeric_summary(array: Any) -> dict[str, Any]:
        """Extract numeric statistics from array"""
        result: dict[str, Any] = {}

        if np.issubdtype(array.dtype, np.number):
            finite_values = array[np.isfinite(array)]
            result = {
                "finite_count": int(finite_values.size),
                "nan_count": int(np.isnan(array).sum()),
                "infinite_count": int(np.isinf(array).sum()),
                "minimum": float(finite_values.min()) if finite_values.size else None,
                "maximum": float(finite_values.max()) if finite_values.size else None,
                "mean": float(finite_values.mean()) if finite_values.size else None,
                "standard_deviation": float(finite_values.std())
                if finite_values.size > 1
                else 0.0,
            }

        return result

    @staticmethod
    def summarize_value(value: Any) -> dict[str, Any]:
        """Create JSON-safe summary of fitted value (mean, std, etc)"""
        try:
            import xarray as xr
        except ImportError:
            xr = None

        if isinstance(value, torch.Tensor):
            return {"type": "torch.Tensor", **PreprocessingUtils.tensor_summary(value)}

        if xr is not None and isinstance(value, (xr.Dataset, xr.DataArray)):
            return PreprocessingUtils.array_summary(value)

        if isinstance(value, np.ndarray):
            return PreprocessingUtils.array_summary(value)

        if isinstance(value, (str, int, float, bool)) or value is None:
            return {"type": type(value).__name__, "value": value}

        return {"type": type(value).__name__, "representation": repr(value)[:100]}

    @staticmethod
    def inspect_fitted_pipeline(
        pipeline: Any,
        *,
        source_name: str,
        configured_steps: list[dict[str, Any]],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Inspect and export fitted preprocessing pipeline state"""
        source_dir = output_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)

        result: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "source": source_name,
            "pipeline_type": type(pipeline).__name__,
            "configured_steps": deepcopy(configured_steps),
            "fitted_preprocessors": [],
        }

        # Inspect each fitted preprocessor in the pipeline
        if hasattr(pipeline, "preprocessors"):
            for idx, (step_config, fitted_preprocessor) in enumerate(
                zip(configured_steps, pipeline.preprocessors)
            ):
                preprocessor_name = step_config.get("name", f"preprocessor_{idx}")

                preprocessor_info: dict[str, Any] = {
                    "index": idx,
                    "name": preprocessor_name,
                    "class": type(fitted_preprocessor).__name__,
                    "configured_args": step_config.get("args", {}),
                    "fitted_attributes": {},
                }

                # Extract fitted statistics (mean_, std_, etc)
                for attr_name in dir(fitted_preprocessor):
                    if attr_name.endswith("_") and not attr_name.startswith("_"):
                        attr_value = getattr(fitted_preprocessor, attr_name, None)
                        if attr_value is not None:
                            preprocessor_info["fitted_attributes"][attr_name] = (
                                PreprocessingUtils.summarize_value(attr_value)
                            )

                result["fitted_preprocessors"].append(preprocessor_info)

        # Save detailed report
        report_path = source_dir / "fitted_pipeline_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True, default=str)

        return result

    @staticmethod
    def find_joblib_files(case_dir: Path) -> list[Path]:
        """Find all .joblib files under case directory (excluding subdirs)"""
        joblib_files = []
        for path in case_dir.glob("*.joblib"):
            if path.is_file():
                joblib_files.append(path.resolve())
        # Also check preprocessing_pipeline subdirectory
        for path in (case_dir / "preprocessing_pipeline").glob("*.joblib"):
            if path.is_file():
                joblib_files.append(path.resolve())
        return sorted(joblib_files)

    @staticmethod
    def classify_joblib_source(path: Path) -> str | None:
        """Classify joblib by filename pattern"""
        name = path.name.lower()
        if "observation" in name or "obs" in name:
            return "observation"
        if "condition" in name:
            return "condition"
        if "model" in name:
            return "model"
        return None

    @staticmethod
    def find_source_joblib(case_dir: Path, *, source_name: str) -> Path | None:
        """Find joblib file for a specific source (model, observation, condition)"""
        all_joblobs = PreprocessingUtils.find_joblib_files(case_dir)

        # Try exact match first
        source_matches = [
            p
            for p in all_joblobs
            if PreprocessingUtils.classify_joblib_source(p) == source_name
        ]

        if len(source_matches) == 1:
            return source_matches[0]

        # If only one joblib exists, assume it's for this source
        if len(all_joblobs) == 1:
            return all_joblobs[0]

        # Ambiguous
        if len(source_matches) > 1:
            raise AssertionError(
                f"Multiple joblib files for {source_name}: "
                f"{[p.name for p in source_matches]}"
            )

        return None

    @staticmethod
    def copy_joblib(joblib_path: Path, *, output_dir: Path) -> Path:
        """Copy joblib file to output directory"""
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / joblib_path.name
        if joblib_path.resolve() != destination.resolve():
            shutil.copy2(joblib_path, destination)
        return destination

    @staticmethod
    def load_and_inspect_joblib(
        joblib_path: Path,
        *,
        source_name: str,
        configured_steps: list[dict[str, Any]],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Load joblib and generate inspection report"""
        pipeline = joblib.load(joblib_path)
        if pipeline is None:
            raise AssertionError(f"joblib.load returned None for {joblib_path}")

        return PreprocessingUtils.inspect_fitted_pipeline(
            pipeline,
            source_name=source_name,
            configured_steps=configured_steps,
            output_dir=output_dir,
        )


class OutputFetcher:
    """Generic output fetcher for any scenario"""

    def __init__(self, output_dir: Path, max_plot_channels: int = 16):
        self.output_dir = output_dir
        self.plot_dir = output_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir = output_dir / "raw_data"
        self.max_plot_channels = max_plot_channels

    def save_batch_metadata(
        self,
        metadata_list: list[Any],
        *,
        output_path: Path,
        batch_sizes: Optional[list[int]] = None,
        batch_index: int = 0,
    ) -> None:
        """Save batch metadata to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if batch_sizes is None:
            batch_sizes = [len(metadata_list)]

        export = {
            "timestamp": datetime.now().isoformat(),
            "batch_index": batch_index,
            "num_batches": len(batch_sizes),
            "batch_sizes": batch_sizes,
            "total_samples": sum(batch_sizes),
            "num_samples": len(metadata_list),
            "samples": [],
            "statistics": {},
        }

        for idx, item in enumerate(metadata_list):
            export["samples"].append(
                {
                    "sample_index": idx,
                    "metadata": item,
                }
            )

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(export, file, indent=2, sort_keys=True, default=str)

    def save_batch_masks(
        self,
        batch: Any,
        *,
        batch_index: int,
    ) -> None:
        """Save masks from batch as NumPy and PyTorch files"""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        mask_fields = ["input_mask", "target_mask", "spatial_mask"]
        masks_metadata = {
            "timestamp": datetime.now().isoformat(),
            "batch_index": batch_index,
            "masks": {},
        }

        for field_name in mask_fields:
            if isinstance(batch, dict):
                mask = batch.get(field_name)
            else:
                mask = getattr(batch, field_name, None)

            if mask is None:
                continue

            # Convert to numpy if it's a tensor
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            elif isinstance(mask, np.ndarray):
                mask_np = mask
            else:
                continue

            # Save as numpy file
            np_output = self.raw_data_dir / f"{field_name}.npy"
            np.save(np_output, mask_np)

            # Also save as PyTorch file if it was a tensor
            if isinstance(mask, torch.Tensor):
                pt_output = self.raw_data_dir / f"{field_name}.pt"
                torch.save(mask, pt_output)

            # Record statistics
            masks_metadata["masks"][field_name] = {
                "shape": tuple(mask_np.shape),
                "dtype": str(mask_np.dtype),
                "saved_as_npy": str(np_output),
                "saved_as_pt": str(self.raw_data_dir / f"{field_name}.pt")
                if isinstance(mask, torch.Tensor)
                else None,
                "analysis": TensorUtils.analyze_mask(mask, name=field_name),
            }

        # Save metadata JSON
        metadata_output = self.raw_data_dir / "masks_metadata.json"
        with metadata_output.open("w") as f:
            json.dump(masks_metadata, f, indent=2, default=str)

    def export_masks_and_data(
        self,
        batch: Any,
        *,
        metadata: list[Any],
        batch_index: int,
    ) -> None:
        """Export masks and data tensors together"""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "batch_index": batch_index,
            "num_samples": len(metadata),
            "samples_metadata": [
                {key: TensorUtils.canonical_value(v) for key, v in item.items()}
                for item in metadata
            ],
            "data_fields": {},
            "mask_fields": {},
        }

        # Save data tensors
        data_fields = ["input", "target", "added_features"]
        for field_name in data_fields:
            if isinstance(batch, dict):
                data = batch.get(field_name)
            else:
                data = getattr(batch, field_name, None)

            if data is None:
                continue

            # Convert to numpy if tensor
            if isinstance(data, torch.Tensor):
                data_np = data.detach().cpu().numpy()
            elif isinstance(data, np.ndarray):
                data_np = data
            else:
                continue

            # Save as numpy file
            np_output = self.raw_data_dir / f"{field_name}_data.npy"
            np.save(np_output, data_np)

            # Also save as PyTorch file if it was a tensor
            if isinstance(data, torch.Tensor):
                pt_output = self.raw_data_dir / f"{field_name}_data.pt"
                torch.save(data, pt_output)

            # Record in metadata
            export_data["data_fields"][field_name] = {
                "shape": tuple(data_np.shape),
                "dtype": str(data_np.dtype),
                "min": float(np.nanmin(data_np)) if data_np.size > 0 else None,
                "max": float(np.nanmax(data_np)) if data_np.size > 0 else None,
                "mean": float(np.nanmean(data_np)) if data_np.size > 0 else None,
                "nan_count": int(np.isnan(data_np).sum()),
                "saved_as_npy": str(np_output),
                "saved_as_pt": str(self.raw_data_dir / f"{field_name}_data.pt")
                if isinstance(data, torch.Tensor)
                else None,
            }

        # Save mask tensors
        mask_fields = ["input_mask", "target_mask", "spatial_mask"]
        for field_name in mask_fields:
            if isinstance(batch, dict):
                mask = batch.get(field_name)
            else:
                mask = getattr(batch, field_name, None)

            if mask is None:
                continue

            # Convert to numpy if tensor
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            elif isinstance(mask, np.ndarray):
                mask_np = mask
            else:
                continue

            # Save as numpy file
            np_output = self.raw_data_dir / f"{field_name}.npy"
            np.save(np_output, mask_np)

            # Also save as PyTorch file if it was a tensor
            if isinstance(mask, torch.Tensor):
                pt_output = self.raw_data_dir / f"{field_name}.pt"
                torch.save(mask, pt_output)

            # Record in metadata
            export_data["mask_fields"][field_name] = {
                "shape": tuple(mask_np.shape),
                "dtype": str(mask_np.dtype),
                "is_binary": bool(np.all(np.isin(mask_np, [0, 1]))),
                "valid_count": int(np.sum(mask_np == 1)),
                "valid_percentage": round(
                    100.0 * np.sum(mask_np == 1) / mask_np.size, 2
                )
                if mask_np.size > 0
                else 0,
                "saved_as_npy": str(np_output),
                "saved_as_pt": str(self.raw_data_dir / f"{field_name}.pt")
                if isinstance(mask, torch.Tensor)
                else None,
            }

        # Save comprehensive metadata JSON
        metadata_output = self.raw_data_dir / "export_metadata.json"
        with metadata_output.open("w") as f:
            json.dump(export_data, f, indent=2, default=str)

    def extract_and_save_premade_mask(self, dataset: Any) -> dict[str, Any]:
        """Extract the premade mask from dataset and save it"""
        masks_dir = self.raw_data_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "premade_mask_extracted": False,
            "premade_mask_file": None,
            "mask_metadata": {},
        }

        if not hasattr(dataset, "mask") or dataset.mask is None:
            results["error"] = "No premade mask in dataset"
            return results

        try:
            mask_data = dataset.mask

            # Convert xarray to numpy if needed
            if hasattr(mask_data, "values"):  # xarray.DataArray
                mask_np = mask_data.values
                results["mask_metadata"]["type"] = "xarray.DataArray"
                results["mask_metadata"]["dims"] = (
                    str(mask_data.dims) if hasattr(mask_data, "dims") else None
                )
                results["mask_metadata"]["coords"] = (
                    list(mask_data.coords.keys())
                    if hasattr(mask_data, "coords")
                    else None
                )
            else:
                mask_np = np.asarray(mask_data)
                results["mask_metadata"]["type"] = type(mask_data).__name__

            # Save as numpy
            np_output = masks_dir / "premade_mask.npy"
            np.save(np_output, mask_np)
            results["premade_mask_file"] = str(np_output)
            results["premade_mask_extracted"] = True

            # Save metadata
            results["mask_metadata"].update(
                {
                    "shape": tuple(mask_np.shape),
                    "dtype": str(mask_np.dtype),
                    "min": float(np.nanmin(mask_np)) if mask_np.size > 0 else None,
                    "max": float(np.nanmax(mask_np)) if mask_np.size > 0 else None,
                    "mean": float(np.nanmean(mask_np)) if mask_np.size > 0 else None,
                    "is_binary": bool(np.all(np.isin(mask_np, [0, 1]))),
                }
            )

            if results["mask_metadata"]["is_binary"]:
                results["mask_metadata"]["valid_count"] = int(np.sum(mask_np == 1))
                results["mask_metadata"]["valid_percentage"] = round(
                    100.0 * np.sum(mask_np == 1) / mask_np.size, 2
                )

            # Save metadata JSON
            metadata_output = masks_dir / "premade_mask_metadata.json"
            with metadata_output.open("w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "premade_mask_metadata": results["mask_metadata"],
                    },
                    f,
                    indent=2,
                    default=str,
                )

            return results

        except Exception as e:
            results["error"] = f"{type(e).__name__}: {str(e)}"
            return results

    def plot_premade_mask(
        self, dataset: Any, lead_times: Optional[list[int]] = None
    ) -> bool:
        """Plot the premade mask if available, using lead_times for x-axis labels

        Parameters
        ----------
        dataset : Any
            Dataset with premade mask
        lead_times : list[int], optional
            Lead times to use for x-axis. If None, tries to extract from dataset config.
        """
        masks_dir = self.raw_data_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        if not hasattr(dataset, "mask") or dataset.mask is None:
            return False

        try:
            mask_data = dataset.mask

            # Convert to numpy
            if hasattr(mask_data, "values"):
                mask_np = mask_data.values
            else:
                mask_np = np.asarray(mask_data)

            # Squeeze to 2D for visualization
            while mask_np.ndim > 2:
                mask_np = mask_np[0]

            if mask_np.ndim != 2:
                return False

            # Try to extract lead_times if not provided
            if lead_times is None:
                # Try from dataset config
                if hasattr(dataset, "config") and isinstance(dataset.config, dict):
                    lead_times_cfg = dataset.config.get("lead_times")
                    if isinstance(lead_times_cfg, np.ndarray):
                        # It's a numpy array
                        lead_times = lead_times_cfg.tolist()
                    elif isinstance(lead_times_cfg, (list, tuple)):
                        # It's a list or tuple
                        lead_times = list(lead_times_cfg)
                    elif isinstance(lead_times_cfg, dict):
                        if "list_lead_times" in lead_times_cfg:
                            lt = lead_times_cfg["list_lead_times"]
                            if isinstance(lt, np.ndarray):
                                lead_times = lt.tolist()
                            else:
                                lead_times = list(lt)
                        elif "start" in lead_times_cfg and "end" in lead_times_cfg:
                            lead_times = list(
                                range(
                                    lead_times_cfg["start"], lead_times_cfg["end"] + 1
                                )
                            )

                # Try from dataset attribute
                if lead_times is None and hasattr(dataset, "lead_times"):
                    lt = dataset.lead_times
                    if isinstance(lt, np.ndarray):
                        lead_times = lt.tolist()
                    elif isinstance(lt, (list, tuple)):
                        lead_times = list(lt)
                    elif hasattr(lt, "__iter__"):
                        try:
                            lead_times = list(lt)
                        except (TypeError, ValueError):
                            pass

            # Create plot
            figure, axes = plt.subplots(1, 1, figsize=(10, 8))
            im = axes.imshow(mask_np, cmap="binary", origin="lower", aspect="auto")

            axes.set_title("Premade Mask from Dataset\n(white=valid, black=masked)")
            axes.set_ylabel("Y (spatial)")

            # Set x-axis to lead times if available
            if lead_times is not None:
                axes.set_xlabel("Lead Time")
                mask_width = mask_np.shape[1]

                if len(lead_times) == mask_width:
                    # Perfect match - show lead times directly
                    step = max(1, len(lead_times) // 10)
                    tick_indices = range(0, len(lead_times), step)
                    axes.set_xticks(tick_indices)
                    axes.set_xticklabels([str(lead_times[i]) for i in tick_indices])
                else:
                    # Shape mismatch - scale lead times to mask width
                    step = max(1, mask_width // 10)
                    tick_indices = range(0, mask_width, step)

                    # Map mask indices to lead times proportionally
                    scaled_labels = []
                    for idx in tick_indices:
                        # Scale index from [0, mask_width) to [0, len(lead_times))
                        lead_idx = int(idx * len(lead_times) / mask_width)
                        lead_idx = min(lead_idx, len(lead_times) - 1)
                        scaled_labels.append(str(lead_times[lead_idx]))

                    axes.set_xticks(tick_indices)
                    axes.set_xticklabels(scaled_labels)

                    # Add info text
                    axes.text(
                        0.5,
                        -0.15,
                        f"Lead times ({len(lead_times)}) scaled to mask width ({mask_width})",
                        transform=axes.transAxes,
                        fontsize=8,
                        ha="center",
                        style="italic",
                    )
            else:
                axes.set_xlabel("X (lead time index)")

            cbar = figure.colorbar(im, ax=axes, shrink=0.8)
            cbar.set_label("Valid (1) / Masked (0)")

            # Add statistics
            is_binary = bool(np.all(np.isin(mask_np, [0, 1])))
            if is_binary:
                valid_pct = 100.0 * np.sum(mask_np == 1) / mask_np.size
            else:
                valid_pct = 100.0 * np.sum(mask_np > 0.5) / mask_np.size

            axes.text(
                0.02,
                0.98,
                f"Valid: {valid_pct:.1f}%\nMasked: {100 - valid_pct:.1f}%",
                transform=axes.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            figure.tight_layout()
            output_file = masks_dir / "premade_mask.png"
            figure.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close(figure)

            return True

        except Exception as e:
            print(f"Error plotting premade mask: {e}")
            return False

    def compare_premade_vs_nan_masks(self, dataset: Any, batch: Any) -> dict[str, Any]:
        """Compare premade mask (from dataset) with NaN-derived masks (from batch)"""
        masks_dir = self.raw_data_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "premade_mask_present": False,
            "nan_derived_masks_present": False,
            "masks_match": None,
            "comparison": {},
        }

        # Get premade mask
        if not hasattr(dataset, "mask") or dataset.mask is None:
            return results

        results["premade_mask_present"] = True

        try:
            premade = dataset.mask
            if hasattr(premade, "values"):
                premade_np = premade.values
            else:
                premade_np = np.asarray(premade)
        except Exception:
            return results

        # Get NaN-derived masks
        if isinstance(batch, dict):
            input_mask = batch.get("input_mask")
            input_data = batch.get("input")
        else:
            input_mask = getattr(batch, "input_mask", None)
            input_data = getattr(batch, "input", None)

        if input_mask is None or input_data is None:
            return results

        results["nan_derived_masks_present"] = True

        try:
            # Convert input_mask to numpy
            if isinstance(input_mask, torch.Tensor):
                input_mask_np = input_mask.detach().cpu().numpy()
            else:
                input_mask_np = np.asarray(input_mask)

            # Reduce dimensions for comparison
            premade_2d = premade_np
            while premade_2d.ndim > 2:
                premade_2d = premade_2d[0]

            input_mask_2d = input_mask_np
            while input_mask_2d.ndim > 2:
                input_mask_2d = input_mask_2d[0]

            # Try to compare
            if premade_2d.shape == input_mask_2d.shape:
                premade_bool = (premade_2d > 0.5).astype(bool)
                input_mask_bool = (input_mask_2d > 0.5).astype(bool)

                matches = np.sum(premade_bool == input_mask_bool)
                total = premade_bool.size

                results["masks_match"] = bool(np.all(premade_bool == input_mask_bool))
                results["comparison"] = {
                    "premade_shape": tuple(premade_np.shape),
                    "input_mask_shape": tuple(input_mask_np.shape),
                    "comparison_shape": tuple(premade_2d.shape),
                    "matching_elements": int(matches),
                    "total_elements": int(total),
                    "match_percentage": round(100.0 * matches / total, 2),
                }
            else:
                results["comparison"] = {
                    "premade_shape": tuple(premade_np.shape),
                    "input_mask_shape": tuple(input_mask_np.shape),
                    "error": "Shapes don't match for direct comparison",
                }

        except Exception as e:
            results["comparison"]["error"] = f"{type(e).__name__}: {str(e)}"

        return results

    def plot_tensor_channels(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
    ) -> None:
        """Plot tensor channels - handles all tensor shapes correctly"""
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
            return

        num_channels = tensor.shape[1] if tensor.ndim >= 2 else 0
        if num_channels == 0:
            return

        channels_to_plot = min(num_channels, self.max_plot_channels)

        for ch_idx in range(channels_to_plot):
            try:
                value = tensor.detach().float().cpu()

                # Handle different tensor shapes consistently
                if value.ndim >= 4:
                    value = value[0, ch_idx]  # (B, C, H, W) → (H, W)
                elif value.ndim == 3:
                    value = value[ch_idx]  # (C, H, W) → (H, W)
                elif value.ndim == 2:
                    # NEW: Handle 2D tensors (B, features) → reshape to (B, features) as image
                    value = value[:, ch_idx : ch_idx + 1]  # Extract column as 2D image
                else:
                    continue

                # Squeeze remaining extra dimensions
                while value.ndim > 2:
                    value = value[0]

                if value.ndim != 2:
                    continue

                image = value.numpy()

                figure, axes = plt.subplots(1, 1, figsize=(6, 5))
                im = axes.imshow(image, origin="lower", aspect="auto", cmap="viridis")
                axes.set_title(f"{name} - Channel {ch_idx}")
                figure.colorbar(im, ax=axes, shrink=0.8)
                figure.tight_layout()

                output_file = (
                    self.plot_dir
                    / f"{StringUtils.clean_name(name)}_channel_{ch_idx:03d}.png"
                )
                figure.savefig(output_file, dpi=150, bbox_inches="tight")
                plt.close(figure)

            except (IndexError, ValueError, RuntimeError):
                continue

    def plot_tensor_channels_unflattened(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
        spatial_shape: tuple[int, int] = (60, 120),
    ) -> None:
        """
        Plot tensor channels from flattened data by reshaping to 2D spatial grid.

        Handles shapes:
        - (B, C, flattened) where flattened = H*W
        """
        print(
            f"DEBUG plot_tensor_channels_unflattened START: name={name}, tensor.shape={tensor.shape}, spatial_shape={spatial_shape}"
        )

        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
            print("DEBUG: Early return - invalid tensor")
            return

        num_channels = tensor.shape[1] if tensor.ndim >= 2 else 0
        print(f"DEBUG: num_channels={num_channels}")

        if num_channels == 0:
            print("DEBUG: Early return - num_channels is 0")
            return

        channels_to_plot = min(num_channels, self.max_plot_channels)
        print(f"DEBUG: channels_to_plot={channels_to_plot}")

        # Reshape from (B, C, flattened) to (B, C, H, W) if needed
        tensor_to_plot = tensor
        if tensor.ndim == 3:
            batch_size, num_ch, flattened = tensor.shape
            expected_flattened = spatial_shape[0] * spatial_shape[1]

            print(f"DEBUG: Before reshape: tensor.shape={tensor.shape}")
            print(
                f"DEBUG: batch_size={batch_size}, num_ch={num_ch}, flattened={flattened}"
            )
            print(f"DEBUG: expected_flattened={expected_flattened}")

            if flattened == expected_flattened:
                tensor_to_plot = tensor.reshape(
                    batch_size, num_ch, spatial_shape[0], spatial_shape[1]
                )
                print(
                    f"DEBUG: After reshape: tensor_to_plot.shape={tensor_to_plot.shape}"
                )
            else:
                print(
                    f"DEBUG: Flattened size mismatch! {flattened} != {expected_flattened}"
                )

        print(f"DEBUG: About to loop over {channels_to_plot} channels")

        for ch_idx in range(channels_to_plot):
            print(f"DEBUG: Processing channel {ch_idx}")

            try:
                value = tensor_to_plot.detach().float().cpu()
                print(f"DEBUG: value.shape={value.shape}, value.ndim={value.ndim}")

                # Extract batch 0, channel ch_idx
                if value.ndim >= 4:
                    print(f"DEBUG: Extracting from 4D tensor: value[0, {ch_idx}]")
                    value = value[0, ch_idx]  # (B, C, H, W) → (H, W)
                elif value.ndim == 3:
                    print(f"DEBUG: Extracting from 3D tensor: value[{ch_idx}]")
                    value = value[ch_idx]  # (C, H, W) → (H, W)
                else:
                    print(f"DEBUG: Skipping - value.ndim={value.ndim}")
                    continue

                print(
                    f"DEBUG: After extraction: value.shape={value.shape}, value.ndim={value.ndim}"
                )

                # Squeeze remaining extra dimensions
                while value.ndim > 2:
                    print(f"DEBUG: Squeezing - value.shape={value.shape}")
                    value = value[0]

                print(
                    f"DEBUG: After squeeze: value.shape={value.shape}, value.ndim={value.ndim}"
                )

                if value.ndim != 2:
                    print(f"DEBUG: Skipping - final ndim={value.ndim} != 2")
                    continue

                image = value.numpy()
                print(
                    f"DEBUG: image.shape={image.shape}, min={np.nanmin(image):.4f}, max={np.nanmax(image):.4f}"
                )

                figure, axes = plt.subplots(1, 1, figsize=(10, 8))
                im = axes.imshow(image, origin="lower", aspect="auto", cmap="viridis")
                axes.set_title(f"{name}, channel {ch_idx}")

                # Set proper spatial axis labels
                height, width = image.shape
                axes.set_xlabel("X (longitude)")
                axes.set_ylabel("Y (latitude)")

                # Set ticks to match spatial dimensions
                axes.set_xticks([0, width // 4, width // 2, 3 * width // 4, width - 1])
                axes.set_xticklabels(
                    [0, width // 4, width // 2, 3 * width // 4, width - 1]
                )
                axes.set_yticks(
                    [0, height // 4, height // 2, 3 * height // 4, height - 1]
                )
                axes.set_yticklabels(
                    [0, height // 4, height // 2, 3 * height // 4, height - 1]
                )

                cbar = figure.colorbar(im, ax=axes, shrink=0.8)
                cbar.set_label("Value")

                figure.tight_layout()

                output_file = (
                    self.plot_dir
                    / f"{StringUtils.clean_name(name)}_channel_{ch_idx:03d}.png"
                )
                print(f"DEBUG: Saving to {output_file}")
                figure.savefig(output_file, dpi=150, bbox_inches="tight")
                plt.close(figure)

            except (IndexError, ValueError, RuntimeError) as e:
                print(f"DEBUG: Error in channel {ch_idx}: {e}")
                continue

        print(f"DEBUG plot_tensor_channels_unflattened END: name={name}")

    def plot_mask(
        self,
        mask: torch.Tensor,
        *,
        name: str,
        lead_times: Optional[list[int]] = None,
    ) -> None:
        """
        Plot binary mask with optional lead times on x-axis.

        Parameters
        ----------
        mask : torch.Tensor
            Mask tensor, typically shape (1, flattened) or (H, W)
        name : str
            Name of the mask for plot title
        lead_times : list[int], optional
            Lead times to use as x-axis labels
        """
        if mask is None or not isinstance(mask, torch.Tensor):
            return

        mask_np = mask.detach().cpu().numpy()

        while mask_np.ndim > 2:
            mask_np = mask_np[0]

        if mask_np.ndim != 2:
            return

        figure, axes = plt.subplots(1, 1, figsize=(12, 6))
        im = axes.imshow(mask_np, cmap="binary", origin="lower", aspect="auto")

        axes.set_title(f"{name} - Mask")
        axes.set_ylabel("Y")

        # Set x-axis to lead times if provided
        if lead_times is not None and len(lead_times) == mask_np.shape[1]:
            axes.set_xlabel("Lead Time")
            axes.set_xticks(range(0, len(lead_times), max(1, len(lead_times) // 10)))
            axes.set_xticklabels(
                [
                    str(lead_times[i])
                    for i in range(0, len(lead_times), max(1, len(lead_times) // 10))
                ]
            )
        else:
            axes.set_xlabel("X")
            if lead_times is not None:
                axes.text(
                    0.5,
                    -0.15,
                    f"Lead times available but shape mismatch: {len(lead_times)} times vs {mask_np.shape[1]} width",
                    transform=axes.transAxes,
                    fontsize=8,
                    ha="center",
                    style="italic",
                )

        cbar = figure.colorbar(im, ax=axes, shrink=0.8)
        cbar.set_label("Valid (1) / Masked (0)")

        figure.tight_layout()

        output_file = self.plot_dir / f"{StringUtils.clean_name(name)}_mask.png"
        figure.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(figure)

    def plot_added_features(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
    ) -> None:
        """
        Plot added features with proper sample index labeling.
        """
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            return

        batch_size, num_features = tensor.shape
        if num_features == 0:
            return

        value = tensor.detach().float().cpu().numpy()
        features_to_plot = min(num_features, self.max_plot_channels)

        for feat_idx in range(features_to_plot):
            try:
                feature_values = value[:, feat_idx]

                figure, axes = plt.subplots(1, 1, figsize=(10, 6))

                # Create bar plot with sample indices on x-axis
                sample_indices = np.arange(len(feature_values))
                axes.bar(
                    sample_indices,
                    feature_values,
                    color="steelblue",
                    alpha=0.7,
                    edgecolor="navy",
                    width=0.6,  # Slightly narrower bars for better appearance
                )

                axes.set_title(f"{name}, channel {feat_idx:03d}")
                axes.set_xlabel("Sample Index")
                axes.set_ylabel("Value")

                # Set x-axis ticks to show only sample indices (not all intermediate values)
                if batch_size <= 10:
                    axes.set_xticks(sample_indices)
                    axes.set_xticklabels([str(i) for i in sample_indices])
                else:
                    # For larger batches, show fewer ticks
                    num_ticks = min(11, batch_size + 1)
                    tick_positions = np.linspace(
                        0, batch_size - 1, num_ticks, dtype=int
                    )
                    axes.set_xticks(tick_positions)
                    axes.set_xticklabels([str(i) for i in tick_positions])

                # Calculate statistics
                vmin = float(np.nanmin(feature_values))
                vmax = float(np.nanmax(feature_values))
                vmean = float(np.nanmean(feature_values))
                vstd = float(np.nanstd(feature_values))

                stats_text = f"Min: {vmin:.3f}\nMax: {vmax:.3f}\nMean: {vmean:.3f}\nStd: {vstd:.3f}"
                axes.text(
                    0.98,
                    0.97,
                    stats_text,
                    transform=axes.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
                )

                axes.grid(True, alpha=0.3, axis="y", linestyle="--")
                figure.tight_layout()

                output_file = (
                    self.plot_dir
                    / f"{StringUtils.clean_name(name)}_channel_{feat_idx:03d}.png"
                )
                figure.savefig(output_file, dpi=150, bbox_inches="tight")
                plt.close(figure)

            except (IndexError, ValueError, RuntimeError):
                continue

    @staticmethod
    def _detect_spatial_shape(flattened_size: int) -> tuple[int, int] | None:
        """Auto-detect spatial shape (H, W) from flattened size.

        Returns the most reasonable (H, W) pair where H*W = flattened_size,
        preferring square-ish grids close to common climate grid sizes.
        """
        # Known common sizes first
        known_sizes = {
            7200: (60, 120),  # Common climate grid
            8256: (86, 96),  # Observation grids
        }
        if flattened_size in known_sizes:
            return known_sizes[flattened_size]

        # Auto-detect: find all divisors
        divisors = []
        for h in range(1, int(flattened_size**0.5) + 1):
            if flattened_size % h == 0:
                w = flattened_size // h
                # Prefer roughly square-ish grids
                aspect_ratio = max(h, w) / min(h, w)
                divisors.append((aspect_ratio, h, w))

        if not divisors:
            return None

        # Sort by aspect ratio (closest to 1 = most square-ish)
        divisors.sort()
        _, h, w = divisors[0]
        return (h, w)

    def plot_batch(self, batch: Any) -> None:
        """Plot all data and masks from batch - handles both flattened and spatial data"""
        if isinstance(batch, dict):
            input_tensor = batch.get("input")
            target_tensor = batch.get("target")
            added_features = batch.get("added_features")
            temporal_mask = batch.get("temporal_mask")
            spatial_mask = batch.get("spatial_mask")
            input_mask = batch.get("input_mask")
            target_mask = batch.get("target_mask")
            metadata = batch.get("metadata")
        else:
            input_tensor = getattr(batch, "input", None)
            target_tensor = getattr(batch, "target", None)
            added_features = getattr(batch, "added_features", None)
            temporal_mask = getattr(batch, "temporal_mask", None)
            spatial_mask = getattr(batch, "spatial_mask", None)
            input_mask = getattr(batch, "input_mask", None)
            target_mask = getattr(batch, "target_mask", None)
            metadata = getattr(batch, "metadata", None)

        # Extract lead times from metadata if available
        lead_times = None
        if isinstance(metadata, dict) and "lead_time" in metadata:
            if isinstance(metadata.get("lead_time"), (list, tuple)):
                lead_times = list(metadata["lead_time"])
            elif isinstance(metadata.get("lead_time"), (int, float)):
                lead_times = [metadata["lead_time"]]

        # Plot data tensors with auto-detected spatial shapes
        if isinstance(input_tensor, torch.Tensor):
            print(f"DEBUG input: shape={input_tensor.shape}, ndim={input_tensor.ndim}")
            if input_tensor.ndim == 3:
                spatial_shape = self._detect_spatial_shape(input_tensor.shape[2])
                print(f"DEBUG input: detected spatial_shape={spatial_shape}")
                if spatial_shape:
                    print(
                        f"DEBUG input: calling plot_tensor_channels_unflattened with spatial_shape={spatial_shape}"
                    )
                    self.plot_tensor_channels_unflattened(
                        input_tensor, name="input", spatial_shape=spatial_shape
                    )
                else:
                    print("DEBUG input: calling plot_tensor_channels (fallback)")
                    self.plot_tensor_channels(input_tensor, name="input")
            else:
                print("DEBUG input: calling plot_tensor_channels (ndim != 3)")
                self.plot_tensor_channels(input_tensor, name="input")

        if isinstance(target_tensor, torch.Tensor):
            print(
                f"DEBUG target: shape={target_tensor.shape}, ndim={target_tensor.ndim}"
            )
            if target_tensor.ndim == 3:
                spatial_shape = self._detect_spatial_shape(target_tensor.shape[2])
                print(f"DEBUG target: detected spatial_shape={spatial_shape}")
                if spatial_shape:
                    print(
                        f"DEBUG target: calling plot_tensor_channels_unflattened with spatial_shape={spatial_shape}"
                    )
                    self.plot_tensor_channels_unflattened(
                        target_tensor, name="target", spatial_shape=spatial_shape
                    )
                else:
                    print("DEBUG target: calling plot_tensor_channels (fallback)")
                    self.plot_tensor_channels(target_tensor, name="target")
            else:
                print("DEBUG target: calling plot_tensor_channels (ndim != 3)")
                self.plot_tensor_channels(target_tensor, name="target")

        if isinstance(added_features, torch.Tensor):
            self.plot_tensor_channels(added_features, name="added_features")

        # Plot all masks with lead times
        if isinstance(input_mask, torch.Tensor):
            print(f"DEBUG input_mask: shape={input_mask.shape}")
            self.plot_mask(input_mask, name="input_mask", lead_times=lead_times)

        if isinstance(target_mask, torch.Tensor):
            print(f"DEBUG target_mask: shape={target_mask.shape}")
            self.plot_mask(target_mask, name="target_mask", lead_times=lead_times)

        if isinstance(temporal_mask, torch.Tensor):
            print(f"DEBUG temporal_mask: shape={temporal_mask.shape}")
            self.plot_mask(temporal_mask, name="temporal_mask", lead_times=lead_times)

        if isinstance(spatial_mask, torch.Tensor):
            print(f"DEBUG spatial_mask: shape={spatial_mask.shape}")
            self.plot_mask(spatial_mask, name="spatial_mask", lead_times=lead_times)


class ConfigManager:
    """Base config management - override in subclasses"""

    @staticmethod
    def base_config() -> dict[str, Any]:
        """Load base config - OVERRIDE THIS"""
        raise NotImplementedError("Subclass must implement base_config()")

    @staticmethod
    def apply_case_config(
        cfg: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply case params to config - OVERRIDE THIS"""
        raise NotImplementedError("Subclass must implement apply_case_config()")

    @staticmethod
    def write_yaml(cfg: dict[str, Any], output_path: Path) -> Path:
        """Write config to YAML - reusable"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        return output_path


class PathManager:
    """Path generation for test cases"""

    def __init__(self, cases_dir: Path, output_dir: Path):
        self.cases_dir = cases_dir
        self.output_dir = output_dir

    def case_paths(self, case_name: str) -> dict[str, Path]:
        """Return base case directory. Subclasses should extend this."""
        case_dir = self.cases_dir / StringUtils.clean_name(case_name)
        return {
            "case_dir": case_dir,
            "config_path": case_dir / "config.yaml",
            "log_path": case_dir / "case.log",
            "preprocessor_dir": case_dir / "preprocessing_pipeline",
            "plot_dir": case_dir / "plots",
            "metadata_path": case_dir / "batch_metadata.json",
            "raw_data_dir": case_dir / "raw_data",
            "metadata_statistics_path": case_dir / "metadata_statistics.json",
        }

    def prepare_output_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cases_dir.mkdir(parents=True, exist_ok=True)


class ResultsManager:
    """Results tracking and reporting"""

    def __init__(self, results_csv: Path, summary_path: Path, csv_columns: list[str]):
        self.results_csv = results_csv
        self.summary_path = summary_path
        self.csv_columns = csv_columns

    def init_csv(self) -> None:
        self.results_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.results_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.csv_columns,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()

    def log_result(
        self,
        row: dict[str, Any],
    ) -> None:
        with self.results_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.csv_columns,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writerow(row)
            file.flush()

    def write_summary(
        self,
        total_cases: int,
        passed_count: int,
        failed_count: int,
    ) -> None:
        with self.summary_path.open("w", encoding="utf-8") as f:
            f.write(f"timestamp: {datetime.now().isoformat()}\n")
            f.write(f"total_cases: {total_cases}\n")
            f.write(f"passed: {passed_count}\n")
            f.write(f"failed: {failed_count}\n")
            f.write(f"results_csv: {self.results_csv}\n")
            f.write(f"summary_path: {self.summary_path}\n")

    def compute_and_save_metadata_statistics(
        self,
        metadata_list: list[Any],
        output_path: Path,
        init_time_dim: str = "init_time",
        lead_time_dim: str = "lead_time",
    ) -> dict[str, Any]:
        """Compute and save metadata statistics"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not metadata_list:
            stats = {}
        else:
            init_times = []
            lead_times = []

            for item in metadata_list:
                if isinstance(item, dict):
                    init_times.append(item.get(init_time_dim))
                    lead_times.append(item.get(lead_time_dim))

            stats = {
                "total_samples": len(metadata_list),
                "unique_init_times": sorted(
                    set(str(t) for t in init_times if t is not None)
                ),
                "unique_lead_times": sorted(
                    set(str(t) for t in lead_times if t is not None)
                ),
            }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)

        return stats


class DataModulesUtils:
    """Utilities specific to data_modules testing"""

    @staticmethod
    def assert_tensor(name: str, value: Any) -> None:
        """Check that value is a valid tensor"""
        if not isinstance(value, torch.Tensor):
            raise AssertionError(f"{name} is not a tensor: {type(value).__name__}")
        if value.numel() == 0:
            raise AssertionError(f"{name} is empty")
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise AssertionError(f"{name} contains non-finite values")

    @staticmethod
    def validate_masks_in_batch(batch: Any) -> dict[str, Any]:
        """Validate all masks in batch"""
        mask_fields = ["input_mask", "target_mask", "spatial_mask"]
        results = {}

        for field_name in mask_fields:
            if isinstance(batch, dict):
                mask = batch.get(field_name)
            else:
                mask = getattr(batch, field_name, None)

            if mask is not None:
                results[field_name] = TensorUtils.analyze_mask(mask, name=field_name)

        return results

    @staticmethod
    def test_dataset_getitem_format(
        dataset: Any, init_time_dim: str = "init_time", lead_time_dim: str = "lead_time"
    ) -> None:
        """Test that dataset.__getitem__ returns proper format"""
        if len(dataset) <= 0:
            raise AssertionError("Dataset is empty")

        indices = sorted({0, len(dataset) // 2, len(dataset) - 1})

        for index in indices:
            item = dataset[index]

            if not isinstance(item, tuple) or len(item) != 2:
                raise AssertionError(
                    f"__getitem__ must return 2-tuple, got {type(item)}"
                )

            sample_data, metadata = item

            if not isinstance(sample_data, dict):
                raise AssertionError(
                    f"Sample must be dict, got {type(sample_data).__name__}"
                )
            if not isinstance(metadata, dict):
                raise AssertionError(
                    f"Metadata must be dict, got {type(metadata).__name__}"
                )

            if "input" not in sample_data or "target" not in sample_data:
                raise AssertionError(f"Missing input or target at index {index}")

            DataModulesUtils.assert_tensor(
                f"sample[{index}].input", sample_data["input"]
            )
            DataModulesUtils.assert_tensor(
                f"sample[{index}].target", sample_data["target"]
            )

            if "input_mask" in sample_data:
                mask_analysis = TensorUtils.analyze_mask(sample_data["input_mask"])
                if not mask_analysis["present"]:
                    raise AssertionError(f"Invalid input_mask at {index}")

            required_meta = {init_time_dim, lead_time_dim}
            missing = required_meta - set(metadata.keys())
            if missing:
                raise AssertionError(f"Missing metadata: {sorted(missing)}")

        print("Dataset.__getitem__ validated")

    @staticmethod
    def check_premade_mask_in_dataset(dataset: Any) -> dict[str, Any]:
        """Check if the dataset has a premade mask that's not being returned"""
        results = {
            "has_mask_attribute": hasattr(dataset, "mask"),
            "mask_value": None,
            "mask_is_returned": False,
        }

        if hasattr(dataset, "mask") and dataset.mask is not None:
            results["mask_value"] = {
                "type": type(dataset.mask).__name__,
                "shape": str(dataset.mask.shape)
                if hasattr(dataset.mask, "shape")
                else None,
            }

            try:
                sample = dataset[0]
                if isinstance(sample, tuple):
                    sample_dict, _ = sample
                else:
                    sample_dict = sample

                results["mask_is_returned"] = "spatial_mask" in sample_dict
            except Exception as e:
                results["error_checking_return"] = str(e)

        return results


class LimitedLoader:
    """Wrapper that limits batch iteration"""

    def __init__(self, loader: Any, max_batches: int | None) -> None:
        self._loader = loader
        self._max_batches = max_batches

    def __iter__(self) -> Iterator:
        for index, batch in enumerate(self._loader):
            if self._max_batches is not None and index >= self._max_batches:
                break
            yield batch

    def __len__(self) -> int:
        length = len(self._loader)
        return length if self._max_batches is None else min(length, self._max_batches)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)
