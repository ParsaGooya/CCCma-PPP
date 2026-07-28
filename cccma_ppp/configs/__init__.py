import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent

with open(CONFIG_DIR / "configs.yaml") as f:
    data = yaml.safe_load(f)


required_sample_dimensions = tuple(data["required_sample_dimensions"])
optional_sample_dimensions = tuple(data["optional_sample_dimensions"])
supported_NN_dimensions_sorted = tuple(data["supported_NN_dimensions_sorted"])

model_data_allowed_dimensions = frozenset(set(data["model_data_allowed_dimensions"])).union(set(supported_NN_dimensions_sorted))
observation_data_allowed_dimensions = frozenset(set(data["observation_data_allowed_dimensions"])).union(set(supported_NN_dimensions_sorted))
condition_data_allowed_dimensions = frozenset(set(data["condition_data_allowed_dimensions"])).union(set(supported_NN_dimensions_sorted))

model_data_required_dimensions = frozenset(set(data["model_data_required_dimensions"]))
observation_data_required_dimensions = frozenset(set(data["observation_data_required_dimensions"]))
condition_data_required_dimensions = frozenset(set(data["condition_data_required_dimensions"]))
            


