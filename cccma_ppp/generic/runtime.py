import dataclasses
from pathlib import Path


@dataclasses.dataclass
class RuntimeContext:
    """
    Container for global runtime configuration and experiment context.

    Attributes
    ----------
    GLOBAL_EXP_DIR : pathlib.Path or str or None
        Root directory for the experiment.
    GLOBAL_CHECKPOINT_DIR : pathlib.Path or str or None
        Directory where model checkpoints are saved.
    GLOBAL_FIGURES_DIR : pathlib.Path or str or None
        Directory where generated figures are stored.
    GLOBAL_LOG_DIR : pathlib.Path or str or None
        Directory where logs are written.

    INPUT_VAR_METADATA : dict or None
        Metadata associated with input variables.
    TARGET_VAR_METADATA : dict or None
        Metadata associated with target variables.
    """
    
    GLOBAL_EXP_DIR: Path | str | None = None
    GLOBAL_CHECKPOINT_DIR: Path | str | None = None
    GLOBAL_FIGURES_DIR: Path | str | None = None
    GLOBAL_LOG_DIR: Path | str | None = None

    INPUT_VAR_METADATA: dict | None = None
    TARGET_VAR_METADATA: dict | None = None
