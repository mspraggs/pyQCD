cmake_minimum_required (VERSION 2.8.8)
enable_testing()
project (kernels)

add_definitions (-DNDIM=4)

# Initialize CXXFLAGS.
set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
# TODO: Multi-compiler/platform support.

set (SRC_DIR ${PROJECT_BINARY_DIR}/src)
set (INC_DIR ${PROJECT_BINARY_DIR}/include)
set (TEST_DIR ${PROJECT_BINARY_DIR}/tests)

file (GLOB
  test_SRC
  RELATIVE
  ${PROJECT_BINARY_DIR}
  ${TEST_DIR}/*.cpp)

find_package (Boost 1.46.0 REQUIRED COMPONENTS unit_test_framework)

include_directories (${Boost_INCLUDE_DIRS} ${INC_DIR})

foreach ( testsourcefile ${test_SRC} )
  string( REPLACE ".cpp" "" testname ${testsourcefile} )
  add_executable( ${testname} ${testsourcefile} )
  target_link_libraries( ${testname} ${Boost_LIBRARIES} )
  add_test( NAME ${testname} COMMAND ${testname} --log_level=all)
endforeach ( testsourcefile ${test_SRC} )