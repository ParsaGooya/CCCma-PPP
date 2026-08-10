#!/bin/bash

set -e

echo "Clearing previous coverage..."
coverage erase

echo "Running test suite with coverage..."
coverage run --branch -m pytest tests

echo "Generating coverage JSON..."
coverage json -o output/test_suite_analysis/baseline_cov.json

echo "Clearing previous coverage..."
coverage erase

echo "Building test map..."
python scripts/build_test_map_parallel.py

echo "Cleaning test map artifacts"
python scripts/parallel_cleanup.py

echo "Pruning tests..."
python scripts/prune_tests.py

echo "Skipping useless tests..."
python scripts/skip_useless_tests.py

echo "Done."