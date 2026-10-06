import dataclasses
from pathlib import Path


@dataclasses.dataclass
class RuntimeContext:
    """
    Store runtime paths and variable metadata shared across the application.

    Unset output directories are derived from ``GLOBAL_EXP_DIR`` during
    initialization.

    Parameters
    ----------
    GLOBAL_EXP_DIR : Path | str | None
        Root directory for experiment outputs.
    GLOBAL_CHECKPOINT_DIR : Path | str | None
        Directory used to store model checkpoints. If ``None``, defaults to the
        ``checkpoints`` directory under ``GLOBAL_EXP_DIR``.
    GLOBAL_FIGURES_DIR : Path | str | None
        Directory used to store generated figures. If ``None``, defaults to the
        ``figures`` directory under ``GLOBAL_EXP_DIR``.
    GLOBAL_LOG_DIR : Path | str | None
        Directory used to store log files. If ``None``, defaults to the ``logs``
        directory under ``GLOBAL_EXP_DIR``.
    GLOBAL_MONITORING_DIR : Path | str | None
        Directory used to store resource-monitoring outputs. If ``None``,
        defaults to the ``resource_monitoring`` directory under
        ``GLOBAL_EXP_DIR``.
    GLOBAL_OUTPUT_DIR : Path | str | None
        Directory used to store inference outputs. If ``None``, defaults to the
        ``inference`` directory under ``GLOBAL_EXP_DIR``.
    INPUT_VAR_METADATA : dict | None
        Metadata describing the model input variables and preprocessing.
    TARGET_VAR_METADATA : dict | None
        Metadata describing the model target variables and preprocessing.
    """

    GLOBAL_EXP_DIR: Path | str | None = dataclasses.field(default_factory=Path.cwd)
    GLOBAL_CHECKPOINT_DIR: Path | str | None = None
    GLOBAL_FIGURES_DIR: Path | str | None = None
    GLOBAL_LOG_DIR: Path | str | None = None
    GLOBAL_MONITORING_DIR: Path | str | None = None
    GLOBAL_OUTPUT_DIR: Path | str | None = None

    INPUT_VAR_METADATA: dict | None = None
    TARGET_VAR_METADATA: dict | None = None

    def __post_init__(self):
        """
        Resolve default runtime directories from the experiment directory.

        Any runtime directory that is not explicitly configured is assigned its
        corresponding default subdirectory under ``GLOBAL_EXP_DIR``.
        """

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
