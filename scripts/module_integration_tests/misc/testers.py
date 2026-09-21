from __future__ import annotations

from pathlib import Path

import dacite

from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.train.train_configs import TrainConfig, prepare_config

# Set this to your config file
CONFIG_PATH = Path("scripts/module_integration_tests/test/test.yaml")


class DataModulesTester:
    def __init__(self, exp_dir: str, base_config_path: Path):
        self.exp_dir = exp_dir
        self.base_config_path = base_config_path

    def run(self, max_batches: int = 4) -> None:
        # 1. Point the library's runtime context at this config's directory
        RuntimeContext.GLOBAL_EXP_DIR = self.exp_dir

        # 2. Load YAML -> dict -> TrainConfig dataclass (dacite validates here)
        data = prepare_config(self.base_config_path)
        config = dacite.from_dict(
            data_class=TrainConfig,
            data=data,
            config=dacite.Config(strict=True),
        )

        # 3. Distributed setup (required even for single-process runs)
        distributed = Distributed.get_instance()
        config.train_loader.setup_distributed(distributed=distributed)

        # 4. Build the actual dataloader
        loader = config.train_loader.build_train_loader(
            return_metadata=True,
            return_spatial_mask=True,
            shuffle=False,
        )

        # 5. Index the dataset directly (calls the library's __getitem__)
        dataset = loader.dataset
        length = len(dataset)
        print(f"dataset length: {length}")
        if length > 0:
            sample = dataset[0]
            print(f"dataset[0] type: {type(sample)}")

        # 6. Iterate a few batches through the real loader (collate, workers, etc.)
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ = (
                batch.get("input")
                if isinstance(batch, dict)
                else getattr(batch, "input", None)
            )
            shape = tuple(input_.shape) if input_ is not None else None
            print(f"batch {i}: input shape = {shape}")

        print("Run completed without error.")


if __name__ == "__main__":
    tester = DataModulesTester(
        exp_dir="scripts/module_integration_tests/test/validation_output",
        base_config_path=CONFIG_PATH,
    )
    tester.run(max_batches=4)
