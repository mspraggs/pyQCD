cmake_minimum_required (VERSION 2.8)
enable_testing()
project (pyQCD)

list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set (EIGEN_INCLUDE_DIR ${CMAKE_MODULE_PATH}/FindEigen3.cmake)

if (CMAKE_BUILD_TYPE)
  message ("-- Building for build type: ${CMAKE_BUILD_TYPE}")
else ()
  set (CMAKE_BUILD_TYPE Release)
endif()

# Initialize CXXFLAGS.
set (CMAKE_CXX_FLAGS_DEBUG "-g -Wall -std=c++11")
set (CMAKE_CXX_FLAGS_RELEASE "-Wall -std=c++11 -O3 -march=native -mtune=native")
# TODO: Multi-compiler/platform support.

find_package (Eigen3 3.0.5 REQUIRED)

include_directories (
  .
  ${EIGEN3_INCLUDE_DIR}
  )

# Workaround to get CLion to recognise all header files
file (GLOB_RECURSE clion_headers ./*.hpp)
add_custom_target (all_clion SOURCES ${clion_headers})

add_subdirectory (core)
add_subdirectory (utils)
add_subdirectory (benchmarks)
add_subdirectory (tests)
