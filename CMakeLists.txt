cmake_minimum_required(VERSION 3.12)
project(pyqhull)

set(CMAKE_CXX_STANDARD 14)

# Set policy to use newer FindPython instead of deprecated FindPythonLibs
cmake_policy(SET CMP0148 NEW)

find_package(pybind11 REQUIRED)

# Find qhull
find_path(QHULL_INCLUDE_DIR NAMES libqhull/qhull_a.h
    PATHS /home/tingfan/micromamba/envs/convexhull/include)
find_library(QHULL_LIBRARY NAMES qhull_r
    PATHS /home/tingfan/micromamba/envs/convexhull/lib)

if(NOT QHULL_INCLUDE_DIR OR NOT QHULL_LIBRARY)
    message(FATAL_ERROR "qhull library not found. Please install qhull development package.")
endif()

# Include directories
include_directories(${QHULL_INCLUDE_DIR})

# Create pybind11 module
pybind11_add_module(pyqhull src/pyqhull.cpp)

# Link libraries
target_link_libraries(pyqhull PRIVATE ${QHULL_LIBRARY})

# Compiler-specific options
target_compile_definitions(pyqhull PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
