def pytest_addoption(parser):
    parser.addoption(
        "--include-pruned",
        action="store_true",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--include-pruned"):
        return

    items[:] = [
        item
        for item in items
        if not any(m.name == "pruned" for m in item.iter_markers())
    ]
