cmake_minimum_required (VERSION 2.6)
enable_testing()
project (pyQCD)

list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set (EIGEN_INCLUDE_DIR ${CMAKE_MODULE_PATH}/FindEigen3.cmake)

# Initialize CXXFLAGS.
set(CMAKE_CXX_FLAGS "-Wall -g -std=c++11")
# TODO: Multi-compiler/platform support.

set (SRC_DIR .)
set (INC_DIR .)
set (TEST_DIR tests)
set (BENCH_DIR benchmarks)

set (test_SRC
  #${TEST_DIR}/test_heatbath.cpp
  ${TEST_DIR}/test_lattice.cpp
  ${TEST_DIR}/test_layout.cpp
  #${TEST_DIR}/test_random.cpp
  #${TEST_DIR}/test_wilson_action.cpp
)

set (benchmark_SRC
  #${BENCH_DIR}/bench_lattice.cpp
  #${BENCH_DIR}/bench_wilson_action.cpp
)

set (core_SRC
  ${SRC_DIR}/core/comms.cpp
  ${SRC_DIR}/core/layout.cpp
  ${SRC_DIR}/core/detail/layout_helpers.cpp)

set (utils_SRC
  ${SRC_DIR}/utils/random.cpp)

find_package (Eigen3 3.0.5 REQUIRED)
find_package (MPI)

if (MPI_CXX_FOUND)
  set (CMAKE_C_COMPILER ${MPI_C_COMPILER})
  set (CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MPI")
endif()

include_directories (
  ${EIGEN3_INCLUDE_DIR}
  ${MPI_CXX_INCLUDE_PATH}
  ${INC_DIR}
)

# Workaround to get clion to identify header files
file (GLOB_RECURSE clion_all_headers ${INC_DIR}/*.hpp)
add_custom_target(all_clion
  SOURCES ${clion_all_headers}
)

add_library(pyQCDcore SHARED ${core_SRC})
target_link_libraries(pyQCDcore)

add_library(pyQCDutils SHARED ${utils_SRC})
target_link_libraries(pyQCDutils)

foreach ( testsourcefile ${test_SRC} )
  string( REPLACE ".cpp" "" testname ${testsourcefile} )
  get_filename_component( testname ${testname} NAME )
  add_executable( ${testname} ${testsourcefile} )
  target_link_libraries( ${testname} ${MPI_CXX_LIBRARIES} pyQCDutils pyQCDcore )
  add_test( NAME ${testname} COMMAND ${testname} --log_level=all)
endforeach()

foreach ( benchsourcefile ${benchmark_SRC} )
  string( REPLACE ".cpp" "" benchname ${benchsourcefile} )
  get_filename_component( benchname ${benchname} NAME )
  add_executable( ${benchname} ${MPI_CXX_LIBRARIES} ${benchsourcefile} )
  target_link_libraries( ${benchname} pyQCDutils pyQCDcore )
endforeach()
