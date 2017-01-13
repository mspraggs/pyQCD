cmake_minimum_required (VERSION 2.6)
enable_testing()
project (pyQCD)

list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set (EIGEN_INCLUDE_DIR ${CMAKE_MODULE_PATH}/FindEigen3.cmake)

# Initialize CXXFLAGS.
set (CMAKE_CXX_FLAGS "-g -Wall")
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
# TODO: Multi-compiler/platform support.

find_package (Eigen3 3.0.5 REQUIRED)

include_directories (
  .
  ${EIGEN3_INCLUDE_DIR}
  )

add_subdirectory (utils)
add_subdirectory (benchmarks)
add_subdirectory (tests)
