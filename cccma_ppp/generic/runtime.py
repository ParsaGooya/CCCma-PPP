import dataclasses
from pathlib import Path


@dataclasses.dataclass
class RuntimeContext:
    GLOBAL_EXP_DIR: Path | str | None = None
    GLOBAL_CHECKPOINT_DIR: Path | str | None = None
    GLOBAL_FIGURES_DIR: Path | str | None = None
    GLOBAL_LOG_DIR: Path | str | None = None
    GLOBAL_OUTPUT_DIR: Path | str | None = None

    INPUT_VAR_METADATA: dict | None = None
    TARGET_VAR_METADATA: dict | None = None
