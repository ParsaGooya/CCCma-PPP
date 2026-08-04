import dataclasses
from pathlib import Path


@dataclasses.dataclass
class RuntimeContext:
    """
    Document this class.

    Parameters
    ----------
    GLOBAL_EXP_DIR : Path | str | None
        Description not yet provided.
    GLOBAL_CHECKPOINT_DIR : Path | str | None
        Description not yet provided.
    GLOBAL_FIGURES_DIR : Path | str | None
        Description not yet provided.
    GLOBAL_LOG_DIR : Path | str | None
        Description not yet provided.
    GLOBAL_OUTPUT_DIR : Path | str | None
        Description not yet provided.
    INPUT_VAR_METADATA : dict | None
        Description not yet provided.
    TARGET_VAR_METADATA : dict | None
        Description not yet provided.
    """

    GLOBAL_EXP_DIR: Path | str | None = None
    GLOBAL_CHECKPOINT_DIR: Path | str | None = None
    GLOBAL_FIGURES_DIR: Path | str | None = None
    GLOBAL_LOG_DIR: Path | str | None = None
    GLOBAL_OUTPUT_DIR: Path | str | None = None

    INPUT_VAR_METADATA: dict | None = None
    TARGET_VAR_METADATA: dict | None = None
