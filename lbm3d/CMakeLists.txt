cmake_minimum_required(VERSION 3.20)

# 3D extension (D3Q19) of the Lattice Boltzmann solver. Stand alone project:
#   cmake -S lbm3d -B build3d && cmake --build build3d -j
project(lbm3d CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(LBM3D_DOUBLE "Use double instead of single precision" OFF)
option(LBM3D_CUDA "Build the CUDA back end when a CUDA compiler is found" ON)
option(LBM3D_CUDA_MOCK "Build Gpu3D.cu as plain C++ against tests/cuda_mock.hpp (logic test, no GPU needed)" OFF)

add_executable(lbm3d src/Main3D.cpp src/Sim3D.cpp src/Cpu3D.cpp)
target_include_directories(lbm3d PRIVATE include)
target_compile_options(lbm3d PRIVATE -Wall -Wextra)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-march=native LBM3D_HAS_MARCH_NATIVE)
if(LBM3D_HAS_MARCH_NATIVE)
    target_compile_options(lbm3d PRIVATE -march=native)
endif()

if(LBM3D_DOUBLE)
    target_compile_definitions(lbm3d PRIVATE LBM_DOUBLE)
endif()

# OpenMP is optional: without it the same code is the serial version
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lbm3d PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found: lbm3d will run on a single thread")
endif()

if(LBM3D_CUDA_MOCK)
    set_source_files_properties(src/Gpu3D.cu PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-xc++")
    target_sources(lbm3d PRIVATE src/Gpu3D.cu)
    target_compile_definitions(lbm3d PRIVATE LBM3D_USE_CUDA LBM3D_CUDA_MOCK)
    message(STATUS "lbm3d: CUDA back end MOCKED on the host (logic test only)")
elseif(LBM3D_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        # the target was created before CUDA was enabled, so it did not pick this up
        set_target_properties(lbm3d PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
        target_sources(lbm3d PRIVATE src/Gpu3D.cu)
        target_compile_definitions(lbm3d PRIVATE LBM3D_USE_CUDA)
        message(STATUS "lbm3d: CUDA back end enabled")
    else()
        message(STATUS "lbm3d: no CUDA compiler found, CPU only build")
    endif()
endif()
