#!/bin/bash
set -e
echo "Running black check"
poetry run black --check .
echo "Running ruff check"
poetry run ruff check .
