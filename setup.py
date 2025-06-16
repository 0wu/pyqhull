from setuptools import setup, Extension
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from pybind11 import get_include
except ImportError:
    # Fallback for when pybind11 is not yet installed
    from setuptools import Extension as Pybind11Extension
    from distutils.command.build_ext import build_ext

ext_modules = [
    Pybind11Extension(
        "pyqhull",
        [
            "src/pyqhull.cpp",
        ],
        libraries=["qhull_r"],
        cxx_std=14,
    ),
]

setup(
    name="pyqhull",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pybind11"
    ],
)
