import yaml
from pathlib import Path
from typing import Literal

lead_time_unit = Literal["day", "month"]

CONFIG_DIR = Path(__file__).parent

with open(CONFIG_DIR / "configs.yaml") as f:
    data = yaml.safe_load(f)


required_sample_dimensions = tuple(data["required_sample_dimensions"])
lead_time_resolution = data["lead_time_resolution"]
realization_dim = data["realization_dim"]
supported_NN_dimensions_sorted = tuple(data["supported_NN_dimensions_sorted"])

if lead_time_resolution not in ["day", 'month']:
    raise ValueError(
        f"lead_time_resolution must be in {lead_time_resolution}."
    )

model_data_allowed_dimensions = frozenset(
    set(data["model_data_allowed_dimensions"])
).union(set(supported_NN_dimensions_sorted))
observation_data_allowed_dimensions = frozenset(
    set(data["observation_data_allowed_dimensions"])
).union(set(supported_NN_dimensions_sorted))
condition_data_allowed_dimensions = frozenset(
    set(data["condition_data_allowed_dimensions"])
).union(set(supported_NN_dimensions_sorted))

model_data_required_dimensions = frozenset(set(data["model_data_required_dimensions"]))
observation_data_required_dimensions = frozenset(
    set(data["observation_data_required_dimensions"])
)
condition_data_required_dimensions = frozenset(
    set(data["condition_data_required_dimensions"])
)

save_deterministic_guess_only = bool(data["save_deterministic_guess_only"])