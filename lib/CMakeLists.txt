cmake_minimum_required (VERSION 2.8.8)
enable_testing()
project (kernels)

add_definitions (-DNDIM=4)

list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set (EIGEN_INCLUDE_DIR ${CMAKE_MODULE_PATH}/FindEigen3.cmake)

# Initialize CXXFLAGS.
set(CMAKE_CXX_FLAGS "-g -O3 -Wall -std=c++11")
# TODO: Multi-compiler/platform support.

set (SRC_DIR .)
set (INC_DIR .)
set (TEST_DIR tests)

set (test_SRC
  ${TEST_DIR}/test_array.cpp
  ${TEST_DIR}/test_lattice.cpp
  ${TEST_DIR}/test_layout.cpp)

set (utils_SRC
  ${SRC_DIR}/utils/math.cpp)

find_package (Eigen3 3.1.3 REQUIRED)

include_directories (
  ${EIGEN3_INCLUDE_DIR}
  ${INC_DIR}
  )

# Workaround to get clion to identify header files
file (GLOB_RECURSE clion_all_headers ${INC_DIR}/*.hpp)
add_custom_target(all_clion
  SOURCES ${clion_all_headers}
)

add_library(pyQCDutils SHARED ${utils_SRC})
target_link_libraries(pyQCDutils)

foreach ( testsourcefile ${test_SRC} )
  string( REPLACE ".cpp" "" testname ${testsourcefile} )
  get_filename_component( testname ${testname} NAME )
  add_executable( ${testname} ${testsourcefile} )
  target_link_libraries( ${testname} pyQCDutils )
  add_test( NAME ${testname} COMMAND ${testname} --log_level=all)
endforeach ( testsourcefile ${test_SRC} )