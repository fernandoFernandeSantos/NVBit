cmake_minimum_required(VERSION 3.16)
project(record_reg_vals LANGUAGES CXX CUDA)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
enable_language(CUDA)
set(CUDA_HOST_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(
        ../../core
        /usr/local/cuda/include
)
add_library(record_reg_vals inject_funcs.cu record_reg_vals.cu)

set_property(TARGET record_reg_vals PROPERTY CUDA_ARCHITECTURES 35 60 86)

set_target_properties(
        record_reg_vals
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(record_reg_vals LogHelper)