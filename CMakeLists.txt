# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name and C++ standard
project(nn_gp-gpu VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/libraries/include
    ${PROJECT_SOURCE_DIR}/src
)

# Source files
set(SOURCES
    src/main.cpp
    src/model.cpp
    src/gradient_descent.cpp
    src/autoencoder.cpp
    src/utils.cpp
)

# Add executable
add_executable(main ${SOURCES})
