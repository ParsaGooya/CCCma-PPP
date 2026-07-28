from unittest.mock import Mock

import pytest
import torch

import cccma_ppp.data_modules.dataloader as module
from cccma_ppp.data_modules.dataloader import (
    BatchDataABC,
    Dataloader,
    DataloaderConfigABC,
)


class ConcreteBatch(BatchDataABC):
    def __init__(self):
        self.input = torch.tensor([1.0])
        self.target = None
        self.added_features = None

    def to_device(self, device):
        self.input = self.input.to(device)
        return self


class ConcreteConfig(DataloaderConfigABC):
    def __init__(
        self,
        *,
        batch_size=2,
        num_data_workers=0,
        prefetch_factor=7,
        drop_last=False,
        pin_memory=True,
    ):
        self.batch_size = batch_size
        self.num_data_workers = num_data_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.time_features = None

        self.pin_memory = pin_memory
        self.time_features = None
        self.return_spatial_mask = False
        self.reduce_spatial_mask = False

        super().__init__()

    @property
    def available_times(self):
        return [2000, 2001]

    def setup_distributed(self):
        self.setup = True
        return self


class ConcreteDataset(torch.utils.data.Dataset):
    def __init__(self, size=5):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index

    def get_input_shape(self):
        return (2, 3)

    def get_target_shape(self):
        return (1,)

    def get_added_features_dims(self):
        return 4


def collate(
    batch,
    return_spatial_mask=False,
    reduce_spatial_mask=False,
):
    return {
        "items": batch,
        "return_spatial_mask": return_spatial_mask,
        "reduce_spatial_mask": reduce_spatial_mask,
    }


class CapturingLoader:
    instances = []

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs
        self.values = [
            "batch-0",
            "batch-1",
            "batch-2",
        ]
        self.__class__.instances.append(self)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


class CapturingSampler:
    instances = []

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs
        self.epochs = []
        self.__class__.instances.append(self)

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


@pytest.fixture(autouse=True)
def reset_captures():
    CapturingLoader.instances.clear()
    CapturingSampler.instances.clear()


def make_loader(**kwargs):
    return Dataloader(
        config=kwargs.pop(
            "config",
            ConcreteConfig(),
        ),
        dataset=kwargs.pop(
            "dataset",
            ConcreteDataset(),
        ),
        collate_fn=kwargs.pop(
            "collate_fn",
            collate,
        ),
        **kwargs,
    )


@pytest.mark.parametrize(
    (
        "workers",
        "requested_prefetch",
        "expected_prefetch",
    ),
    [
        (0, 9, None),
        (1, 9, 9),
        (4, None, None),
    ],
)
def test_config_init_prefetch_worker_branches(
    workers,
    requested_prefetch,
    expected_prefetch,
):
    config = ConcreteConfig(
        num_data_workers=workers,
        prefetch_factor=requested_prefetch,
    )

    assert config.prefetch_factor is expected_prefetch or (
        config.prefetch_factor == expected_prefetch
    )
    assert config._setup is False
    assert config.pin_memory is False


@pytest.mark.parametrize(
    (
        "world_size",
        "explicit_shuffle",
        "expected_shuffle",
    ),
    [
        (1, None, True),
        (2, None, False),
        (1, False, False),
        (2, True, True),
    ],
)
def test_post_init_shuffle_branches(
    monkeypatch,
    world_size,
    explicit_shuffle,
    expected_shuffle,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )
    monkeypatch.setattr(
        module,
        "DistributedSampler",
        CapturingSampler,
    )

    make_loader(
        world_size=world_size,
        shuffle=explicit_shuffle,
    )

    args = CapturingLoader.instances[-1].kwargs

    assert args["shuffle"] is expected_shuffle


@pytest.mark.parametrize(
    (
        "workers",
        "expected_persistent",
    ),
    [
        (0, False),
        (1, True),
        (3, True),
    ],
)
def test_post_init_worker_persistence_branch(
    monkeypatch,
    workers,
    expected_persistent,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    config = ConcreteConfig(
        num_data_workers=workers,
        prefetch_factor=2,
    )

    make_loader(config=config)

    args = CapturingLoader.instances[-1].kwargs

    assert args["num_workers"] == workers
    assert args["persistent_workers"] is expected_persistent
    assert args["prefetch_factor"] == (None if workers == 0 else 2)

    assert args["pin_memory"] is False


def test_post_init_forwards_spatial_mask_flags(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    config = ConcreteConfig()
    config.return_spatial_mask = True
    config.reduce_spatial_mask = True

    make_loader(config=config)

    wrapped_collate = CapturingLoader.instances[-1].kwargs["collate_fn"]

    result = wrapped_collate([1, 2])

    assert result == {
        "items": [1, 2],
        "return_spatial_mask": True,
        "reduce_spatial_mask": True,
    }


def test_single_process_has_no_sampler(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )
    monkeypatch.setattr(
        module,
        "DistributedSampler",
        CapturingSampler,
    )

    loader = make_loader(world_size=1)

    assert loader.sampler is None
    assert CapturingSampler.instances == []


def test_distributed_sampler_receives_all_arguments(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )
    monkeypatch.setattr(
        module,
        "DistributedSampler",
        CapturingSampler,
    )

    config = ConcreteConfig(
        drop_last=True,
    )

    loader = make_loader(
        config=config,
        world_size=4,
        rank=2,
    )

    sampler = loader.sampler

    assert sampler is CapturingSampler.instances[-1]
    assert sampler.kwargs == {
        "num_replicas": 4,
        "rank": 2,
        "shuffle": True,
        "drop_last": True,
    }


def test_sampler_helper_forwards_extra_kwargs(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )
    monkeypatch.setattr(
        module,
        "DistributedSampler",
        CapturingSampler,
    )

    loader = make_loader(world_size=1)

    loader.world_size = 2
    loader.rank = 1

    sampler = loader._get_dataloader_sampler(
        seed=123,
    )

    assert sampler.kwargs["seed"] == 123
    assert sampler.kwargs["num_replicas"] == 2
    assert sampler.kwargs["rank"] == 1


def test_set_epoch_no_sampler_returns_same_loader(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    loader = make_loader(world_size=1)

    assert loader.set_epoch(7) is loader


def test_set_epoch_distributed_delegates(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )
    monkeypatch.setattr(
        module,
        "DistributedSampler",
        CapturingSampler,
    )

    loader = make_loader(world_size=2)

    assert loader.set_epoch(7) is loader
    assert loader.sampler.epochs == [7]


def test_shape_and_feature_properties_delegate_to_dataset(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    dataset = Mock()
    dataset.get_input_shape.return_value = (8, 9)
    dataset.get_target_shape.return_value = (2,)
    dataset.get_added_features_dim.return_value = 6

    loader = make_loader(dataset=dataset)

    assert loader.input_shape == (8, 9)
    assert loader.target_shape == (2,)
    assert loader.added_features_dim == 6

    dataset.get_input_shape.assert_called_once_with()
    dataset.get_target_shape.assert_called_once_with()
    dataset.get_added_features_dim.assert_called_once_with()


def test_iter_and_len_delegate_to_torch_loader(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    loader = make_loader()

    assert list(loader) == [
        "batch-0",
        "batch-1",
        "batch-2",
    ]
    assert len(loader) == 3


@pytest.mark.parametrize(
    (
        "start",
        "expected",
    ),
    [
        (
            0,
            [
                "batch-0",
                "batch-1",
                "batch-2",
            ],
        ),
        (
            1,
            [
                "batch-1",
                "batch-2",
            ],
        ),
        (3, []),
        (99, []),
    ],
)
def test_subset_loader_start_branches(
    monkeypatch,
    start,
    expected,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    loader = make_loader()

    assert list(loader.subset_loader(start)) == expected


def test_config_available_times_property():
    config = ConcreteConfig()

    assert config.available_times == [2000, 2001]


def test_config_setup_distributed_returns_self():
    config = ConcreteConfig()

    result = config.setup_distributed()

    assert result is config
    assert config.setup is True


def test_config_init_sets_setup_false():
    config = ConcreteConfig()

    assert config._setup is False


def test_config_init_always_sets_pin_memory_false():
    config = ConcreteConfig(
        pin_memory=True,
    )

    assert config.pin_memory is False


def test_config_zero_workers_clears_prefetch_factor():
    config = ConcreteConfig(
        num_data_workers=0,
        prefetch_factor=8,
    )

    assert config.prefetch_factor is None


def test_config_nonzero_workers_preserves_prefetch_factor():
    config = ConcreteConfig(
        num_data_workers=2,
        prefetch_factor=8,
    )

    assert config.prefetch_factor == 8


@pytest.mark.parametrize(
    ("workers", "expected_timeout"),
    [
        (0, 0),
        (1, 60),
        (4, 60),
    ],
)
def test_post_init_worker_timeout_branches(
    monkeypatch,
    workers,
    expected_timeout,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    config = ConcreteConfig(
        num_data_workers=workers,
        prefetch_factor=2,
    )

    make_loader(config=config)

    kwargs = CapturingLoader.instances[-1].kwargs

    assert kwargs["timeout"] == expected_timeout


def test_post_init_forwards_basic_loader_arguments(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    dataset = ConcreteDataset(size=7)
    config = ConcreteConfig(
        batch_size=3,
        num_data_workers=0,
        drop_last=True,
    )

    loader = make_loader(
        config=config,
        dataset=dataset,
    )

    captured = CapturingLoader.instances[-1]

    assert captured.dataset is dataset
    assert captured.kwargs["batch_size"] == 3
    assert captured.kwargs["sampler"] is loader.sampler
    assert captured.kwargs["num_workers"] == 0
    assert captured.kwargs["prefetch_factor"] is None
    assert captured.kwargs["persistent_workers"] is False
    assert captured.kwargs["timeout"] == 0


def test_post_init_forwards_config_pin_memory_value(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    config = ConcreteConfig()
    config.pin_memory = True

    make_loader(config=config)

    kwargs = CapturingLoader.instances[-1].kwargs

    assert kwargs["pin_memory"] is True


def test_post_init_default_spatial_mask_flags(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    config = ConcreteConfig()
    config.return_spatial_mask = False
    config.reduce_spatial_mask = False

    make_loader(config=config)

    wrapped_collate = CapturingLoader.instances[-1].kwargs["collate_fn"]

    result = wrapped_collate([1, 2])

    assert result == {
        "items": [1, 2],
        "return_spatial_mask": False,
        "reduce_spatial_mask": False,
    }


def test_collate_partial_preserves_original_function(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    make_loader()

    wrapped_collate = CapturingLoader.instances[-1].kwargs["collate_fn"]

    assert wrapped_collate.func is collate


def test_collate_partial_contains_configured_flags(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    config = ConcreteConfig()
    config.return_spatial_mask = True
    config.reduce_spatial_mask = False

    make_loader(config=config)

    wrapped_collate = CapturingLoader.instances[-1].kwargs["collate_fn"]

    assert wrapped_collate.keywords == {
        "return_spatial_mask": True,
        "reduce_spatial_mask": False,
    }


def test_distributed_sampler_uses_dataset_instance(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )
    monkeypatch.setattr(
        module,
        "DistributedSampler",
        CapturingSampler,
    )

    dataset = ConcreteDataset()

    loader = make_loader(
        dataset=dataset,
        world_size=2,
    )

    assert loader.sampler.dataset is dataset


def test_sampler_helper_single_process_ignores_extra_kwargs(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )
    monkeypatch.setattr(
        module,
        "DistributedSampler",
        CapturingSampler,
    )

    loader = make_loader(world_size=1)

    sampler = loader._get_dataloader_sampler(
        seed=123,
    )

    assert sampler is None
    assert CapturingSampler.instances == []


def test_set_epoch_accepts_zero_epoch(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )
    monkeypatch.setattr(
        module,
        "DistributedSampler",
        CapturingSampler,
    )

    loader = make_loader(world_size=2)

    result = loader.set_epoch(0)

    assert result is loader
    assert loader.sampler.epochs == [0]


def test_set_epoch_accepts_negative_epoch(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )
    monkeypatch.setattr(
        module,
        "DistributedSampler",
        CapturingSampler,
    )

    loader = make_loader(world_size=2)

    loader.set_epoch(-1)

    assert loader.sampler.epochs == [-1]


def test_subset_loader_default_starts_at_first_batch(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    loader = make_loader()

    assert list(loader.subset_loader()) == [
        "batch-0",
        "batch-1",
        "batch-2",
    ]


def test_iter_returns_underlying_loader_iterator(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    loader = make_loader()

    iterator = iter(loader)

    assert next(iterator) == "batch-0"
    assert next(iterator) == "batch-1"
    assert next(iterator) == "batch-2"

    with pytest.raises(StopIteration):
        next(iterator)


def test_len_returns_zero_for_empty_capturing_loader(
    monkeypatch,
):
    class EmptyLoader(CapturingLoader):
        def __init__(self, dataset, **kwargs):
            super().__init__(dataset, **kwargs)
            self.values = []

    monkeypatch.setattr(
        module,
        "DataLoader",
        EmptyLoader,
    )

    loader = make_loader()

    assert len(loader) == 0
    assert list(loader) == []


def test_shape_properties_return_none_when_dataset_returns_none(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "DataLoader",
        CapturingLoader,
    )

    dataset = Mock()
    dataset.get_input_shape.return_value = None
    dataset.get_target_shape.return_value = None
    dataset.get_added_features_dim.return_value = None

    loader = make_loader(dataset=dataset)

    assert loader.input_shape is None
    assert loader.target_shape is None
    assert loader.added_features_dim is None

    original_input_mask = BatchDataABC._shared_input_mask
    original_target_mask = BatchDataABC._shared_target_mask

    input_mask = torch.ones(1)
    target_mask = torch.zeros(1)

    try:
        ConcreteBatch._shared_input_mask = input_mask
        ConcreteBatch._shared_target_mask = target_mask

        first = ConcreteBatch()
        second = ConcreteBatch()

        assert first._shared_input_mask is input_mask
        assert second._shared_input_mask is input_mask
        assert first._shared_target_mask is target_mask
        assert second._shared_target_mask is target_mask
    finally:
        ConcreteBatch._shared_input_mask = original_input_mask
        ConcreteBatch._shared_target_mask = original_target_mask
