import dataclasses
from pathlib import Path


@dataclasses.dataclass
class RuntimeContext:
    """
    Global runtime configuration for experiment-level settings.

    Parameters
    ----------
    GLOBAL_EXP_DIR : pathlib.Path or str or None, optional
        Root directory for experiment outputs.
    GLOBAL_CHECKPOINT_DIR : pathlib.Path or str or None, optional
        Directory for saving model checkpoints.
    GLOBAL_FIGURES_DIR : pathlib.Path or str or None, optional
        Directory for saving figures and plots.
    GLOBAL_LOG_DIR : pathlib.Path or str or None, optional
        Directory for log files.
    GLOBAL_OUTPUT_DIR : pathlib.Path or str or None, optional
        General output directory for experiment artifacts.

    INPUT_VAR_METADATA : dict or None, optional
        Metadata describing input variables and preprocessing steps.
    TARGET_VAR_METADATA : dict or None, optional
        Metadata describing target variables and preprocessing steps.
    """

    GLOBAL_EXP_DIR: Path | str | None = None
    GLOBAL_CHECKPOINT_DIR: Path | str | None = None
    GLOBAL_FIGURES_DIR: Path | str | None = None
    GLOBAL_LOG_DIR: Path | str | None = None
    GLOBAL_OUTPUT_DIR: Path | str | None = None

    INPUT_VAR_METADATA: dict | None = None
    TARGET_VAR_METADATA: dict | None = None
