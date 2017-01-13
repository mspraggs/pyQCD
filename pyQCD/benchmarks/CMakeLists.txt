cmake_minimum_required (VERSION 2.8)
project(pyQCD)

set (SRC
  bench_lattice.cpp
  bench_wilson_action.cpp)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")

foreach ( benchsourcefile ${SRC} )
  string( REPLACE ".cpp" "" benchname ${benchsourcefile} )
  get_filename_component( benchname ${benchname} NAME )
  add_executable( ${benchname} ${benchsourcefile} )
  target_link_libraries( ${benchname} pyQCDutils )
endforeach()