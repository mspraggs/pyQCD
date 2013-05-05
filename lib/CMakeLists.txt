cmake_minimum_required (VERSION 2.6)
project (pyQCD)

set(SRC_DIR ${PROJECT_BINARY_DIR}/src)
set(INC_DIR ${PROJECT_BINARY_DIR}/include)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(EIGEN_INCLUDE_DIR ${CMAKE_MODULE_PATH}/FindEigen3.cmake)

find_package(PythonLibs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost 1.44.0)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${INC_DIR})

set(SOURCES
  ${SRC_DIR}/pyQCD.cpp
  ${SRC_DIR}/pylattice.cpp
  ${SRC_DIR}/lattice.cpp)

set(CMAKE_SHARED_LIBRARY_PREFIX "")
add_library(pyQCD SHARED ${SRC_DIR}/pyQCD.cpp)