cmake_minimum_required(VERSION 3.18)
project(CUDA_LAB LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Shared library for utilities
file(GLOB COMMON_SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.cu
)
if (COMMON_SOURCES)
    add_library(cuda_common STATIC ${COMMON_SOURCES})
else()
    message(WARNING "No common sources found! Skipping cuda_common library.")
endif()

# Add each benchmark
add_subdirectory(benchmarks/01_vector_add)
