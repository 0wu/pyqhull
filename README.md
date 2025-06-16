# PyQhull - Convex Hull Batch Processing

A Python library that provides efficient batch convex hull computation for 3D points using the qhull library and thread pool parallelization.

## Features

- Batch processing of multiple point clouds
- Thread pool for parallel computation
- Built on the robust qhull library

## Installation

1. Install dependencies:
```bash
# Install qhull development package
sudo apt-get install libqhull-dev  # Ubuntu/Debian
# or
sudo dnf install qhull-devel       # Fedora
# or
micromamba install pybind11 libqhull

# Install Python dependencies
pip install -r requirements.txt
```

2. Build and install:
```bash
pip install .
```

## Usage and Testing

```bash
python test_pyqhull.py
```
