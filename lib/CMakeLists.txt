cmake_minimum_required (VERSION 2.6)
project (pyQCD)

set (SRC_DIR ${PROJECT_BINARY_DIR}/src)
set (INC_DIR ${PROJECT_BINARY_DIR}/include)

file (GLOB
  pyQCD_SRC
  ${SRC_DIR}/*.cpp)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set (EIGEN_INCLUDE_DIR ${CMAKE_MODULE_PATH}/FindEigen3.cmake)

find_package (PythonLibs REQUIRED)
find_package (Eigen3 REQUIRED)
find_package (Boost REQUIRED)
find_package (OpenMP REQUIRED)

if (OPENMP_FOUND)
  message ("OpenMP found")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories (${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PYTHON_INCLUDE_PATH}
  ${INC_DIR})

set (Boost_USE_STATIC_LIBS OFF)
set (Boost_USE_MULTITHREADED ON)
set (Boost_USE_STATIC_RUNTIME OFF)
find_package (Boost COMPONENTS python)
find_package (Boost COMPONENTS random)

set (SOURCES
  ${SRC_DIR}/pyQCD.cpp
  ${SRC_DIR}/pylattice.cpp
  ${SRC_DIR}/lattice.cpp)

set (CMAKE_SHARED_LIBRARY_PREFIX "")
add_library (pyQCD SHARED ${pyQCD_SRC})
target_link_libraries (pyQCD ${Boost_LIBRARIES})

IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_DEFINITIONS("-O3")
ENDIF()
IF(CMAKE_COMPILER_IS_MSVC)
  ADD_DEFINITIONS("/O2")
ENDIF()