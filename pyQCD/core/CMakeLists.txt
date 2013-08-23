cmake_minimum_required (VERSION 2.6)
project (lattice)
enable_testing()

set (TEST_DIR ${PROJECT_BINARY_DIR}/test)
set (SRC_DIR ${PROJECT_BINARY_DIR}/src)
set (INC_DIR ${PROJECT_BINARY_DIR}/include)

file (GLOB
  lattice_SRC
  ${SRC_DIR}/*.cpp)

file (GLOB
  test_SRC
  ${TEST_DIR}/*.cpp
  ${SRC_DIR}/lattice_*.cpp
  ${SRC_DIR}/utils.cpp)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set (EIGEN_INCLUDE_DIR ${CMAKE_MODULE_PATH}/FindEigen3.cmake)

find_package (PythonLibs 2.6 REQUIRED)
find_package (Eigen3 3.1.3 REQUIRED)
find_package (Boost 1.49.0 REQUIRED)
find_package (OpenMP 3.0 REQUIRED)

if (OPENMP_FOUND)
  message ("OpenMP found")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS}
    ${OpenMP_EXE_LINKER_FLAGS})
endif()

include_directories (${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PYTHON_INCLUDE_PATH}
  ${INC_DIR})

set (Boost_USE_STATIC_LIBS OFF)
set (Boost_USE_MULTITHREADED ON)
set (Boost_USE_STATIC_RUNTIME OFF)
find_package (Boost COMPONENTS random python unit_test_framework)

set (CMAKE_SHARED_LIBRARY_PREFIX "")
add_library (lattice SHARED ${lattice_SRC})
target_link_libraries (lattice ${Boost_LIBRARIES})

find_package (Boost COMPONENTS unit_test_framework)

add_definitions(-DBOOST_TEST_DYN_LINK)
add_executable (test_lattice ${test_SRC})
target_link_libraries (test_lattice ${Boost_LIBRARIES})
add_test(lattice test_lattice)

IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_DEFINITIONS("-O3")
ENDIF()
IF(CMAKE_COMPILER_IS_MSVC)
  ADD_DEFINITIONS("/O2")
ENDIF()
