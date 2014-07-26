cmake_minimum_required (VERSION 2.8.8)
enable_testing()
project (kernels)

add_definitions (-DNDIM=4)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set (EIGEN_INCLUDE_DIR ${CMAKE_MODULE_PATH}/FindEigen3.cmake)

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

file (GLOB
  utils_SRC
  ${SRC_DIR}/utils/*.cpp)

find_package (Eigen3 3.1.3 REQUIRED)
find_package (Boost 1.46.0 REQUIRED COMPONENTS unit_test_framework)

include_directories (
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${INC_DIR}
  )

add_library(pyQCDutils SHARED ${utils_SRC})
target_link_libraries(pyQCDutils ${Boost_LIBRARIES})

foreach ( testsourcefile ${test_SRC} )
  string( REPLACE ".cpp" "" testname ${testsourcefile} )
  add_executable( ${testname} ${testsourcefile} )
  target_link_libraries( ${testname} ${Boost_LIBRARIES} pyQCDutils)
  add_test( NAME ${testname} COMMAND ${testname} --log_level=all)
endforeach ( testsourcefile ${test_SRC} )