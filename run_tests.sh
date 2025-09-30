#!/bin/bash
# Convenience script to run colpackpy unit tests

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/ColPack/build/cmake/mywork/lib/shared_library"

echo "=================================="
echo "Running ColPackPy Unit Tests"
echo "=================================="
echo ""
echo "PYTHONPATH: ${PYTHONPATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo ""

# Check if pytest is available
if command -v pytest &> /dev/null; then
    echo "Using pytest..."
    pytest colpackpy/tests/ -v "$@"
else
    echo "pytest not found, using unittest..."
    python -m unittest discover -v colpackpy/tests "$@"
fi
