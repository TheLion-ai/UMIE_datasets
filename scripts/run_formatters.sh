#!/bin/bash
set -e
echo "Running black format"
poetry run black .
echo "Running ruff fix"
poetry run ruff --fix .
