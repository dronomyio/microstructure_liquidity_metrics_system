cmake_minimum_required(VERSION 3.18)
project(LiquidityMetrics LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# SIMD flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
CHECK_CXX_COMPILER_FLAG("-mavx512f" COMPILER_SUPPORTS_AVX512)

if(COMPILER_SUPPORTS_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512bw -mavx512vl")
elseif(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# Source files
set(CPP_SOURCES
    src/cpp/simd_processor.cpp
    src/cpp/polygon_parser.cpp
    src/cpp/liquidity_metrics.cpp
)

set(CUDA_SOURCES
    src/cpp/cuda_processor.cu
)

# CUDA library
cuda_add_library(cuda_metrics SHARED ${CUDA_SOURCES})
target_compile_features(cuda_metrics PUBLIC cuda_std_14)

# Main C++ library
add_library(liquidity_core SHARED ${CPP_SOURCES})
target_link_libraries(liquidity_core 
    cuda_metrics
    OpenMP::OpenMP_CXX
    pthread
)

# Python bindings
pybind11_add_module(liquidity_metrics src/python/bindings.cpp)
target_link_libraries(liquidity_metrics PRIVATE 
    liquidity_core
    cuda_metrics
)

# Main executable
add_executable(liquidity_runner src/cpp/main.cpp)
target_link_libraries(liquidity_runner 
    liquidity_core
    cuda_metrics
)

# Tests
enable_testing()
add_executable(test_simd tests/test_simd.cpp)
target_link_libraries(test_simd liquidity_core)
add_test(NAME TestSIMD COMMAND test_simd)

add_executable(test_cuda tests/test_cuda.cpp)
target_link_libraries(test_cuda cuda_metrics)
add_test(NAME TestCUDA COMMAND test_cuda)
