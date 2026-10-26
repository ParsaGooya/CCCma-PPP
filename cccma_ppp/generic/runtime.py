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
    GLOBAL_MONITORING_DIR : Path | str | None
        Description not yet provided.
    GLOBAL_OUTPUT_DIR : Path | str | None
        Description not yet provided.
    INPUT_VAR_METADATA : dict | None
        Description not yet provided.
    TARGET_VAR_METADATA : dict | None
        Description not yet provided.
    """

    GLOBAL_EXP_DIR: Path | str | None = dataclasses.field(
        default_factory=Path.cwd
    )
    GLOBAL_CHECKPOINT_DIR: Path | str | None = None
    GLOBAL_FIGURES_DIR: Path | str | None = None
    GLOBAL_LOG_DIR: Path | str | None = None
    GLOBAL_MONITORING_DIR: Path | str | None = None
    GLOBAL_OUTPUT_DIR: Path | str | None = None

    INPUT_VAR_METADATA: dict | None = None
    TARGET_VAR_METADATA: dict | None = None

    def __post_init__(self):

        if self.GLOBAL_CHECKPOINT_DIR is None:
            self.GLOBAL_CHECKPOINT_DIR = self.GLOBAL_EXP_DIR / "checkpoints"

        if self.GLOBAL_FIGURES_DIR is None:
            self.GLOBAL_FIGURES_DIR = self.GLOBAL_EXP_DIR / "figures"

        if self.GLOBAL_LOG_DIR is None:
            self.GLOBAL_LOG_DIR = self.GLOBAL_EXP_DIR / "logs"

        if self.GLOBAL_MONITORING_DIR is None:
            self.GLOBAL_MONITORING_DIR = self.GLOBAL_EXP_DIR / "resource_monitoring"

        if self.GLOBAL_OUTPUT_DIR is None:
            self.GLOBAL_OUTPUT_DIR = self.GLOBAL_EXP_DIR / "inference"
        