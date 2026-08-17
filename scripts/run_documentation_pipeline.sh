#!/bin/bash

set -e

echo "Removing old documentation..."
python3 ../scripts/strip_documentation.py

echo "Inserting fresh documentation..."
python3 ../scripts/insert_blank_documentation.py --apply

echo "Done."