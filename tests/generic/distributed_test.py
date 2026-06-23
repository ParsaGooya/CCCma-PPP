import pytest
import torch

import cccma_ppp.generic.distributed as mod


class DummyDist:
    def __init__(self):
        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def is_available(self):
        return True


@pytest.mark.pruned
def test_distributed_false(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    d = mod.Distributed()

    assert d.distributed is False
    assert d.rank == 0
    assert d.world_size == 1
    assert d.is_root()


def test_distributed_true(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    monkeypatch.setattr(torch.cuda, "set_device", lambda x: None)

    monkeypatch.setattr(mod.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(mod.dist, "init_process_group", lambda backend: None)

    d = mod.Distributed()

    assert d.distributed is True
    assert d.rank == 1
    assert d.world_size == 2


def test_singleton(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    mod.Distributed._instance = None

    d1 = mod.Distributed.get_instance()
    d2 = mod.Distributed.get_instance()

    assert d1 is d2


@pytest.mark.pruned
def test_cleanup(monkeypatch):
    monkeypatch.setattr(mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)

    called = {}

    def fake_destroy():
        called["destroy"] = True

    monkeypatch.setattr(mod.dist, "destroy_process_group", fake_destroy)

    d = mod.Distributed()
    d.cleanup()

    assert called.get("destroy", False)


def test_cleanup_not_initialized(monkeypatch):
    monkeypatch.setattr(mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: False)

    d = mod.Distributed()
    d.cleanup()


def test_barrier_called(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    monkeypatch.setattr(torch.cuda, "set_device", lambda x: None)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mod.dist, "init_process_group", lambda backend: None)

    called = {}

    monkeypatch.setattr(mod.dist, "barrier", lambda: called.setdefault("hit", True))

    d = mod.Distributed()
    d.barrier()

    assert called.get("hit", False)


def test_barrier_not_called(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    d = mod.Distributed()
    d.barrier()


def test_all_reduce_called(monkeypatch):
    monkeypatch.setattr(mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)

    called = {}

    def fake_all_reduce(tensor, op):
        called["hit"] = True

    monkeypatch.setattr(mod.dist, "all_reduce", fake_all_reduce)

    d = mod.Distributed()

    t = torch.tensor([1.0])
    d.all_reduce_sum(t)

    assert called.get("hit", False)


def test_all_reduce_not_called(monkeypatch):
    monkeypatch.setattr(mod.dist, "is_available", lambda: False)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: False)

    d = mod.Distributed()
    t = torch.tensor([1.0])

    d.all_reduce_sum(t)


def test_broadcast_called(monkeypatch):
    monkeypatch.setattr(mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)

    called = {}

    def fake_broadcast(tensor, src):
        called["hit"] = True

    monkeypatch.setattr(mod.dist, "broadcast", fake_broadcast)

    d = mod.Distributed()

    t = torch.tensor([1.0])
    d.broadcast(t, src=0)

    assert called.get("hit", False)


def test_broadcast_not_called(monkeypatch):
    monkeypatch.setattr(mod.dist, "is_available", lambda: False)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: False)

    d = mod.Distributed()
    t = torch.tensor([1.0])

    d.broadcast(t)


@pytest.mark.pruned
def test_is_root():
    d = mod.Distributed()
    d.rank = 0
    assert d.is_root()

    d.rank = 1
    assert not d.is_root()