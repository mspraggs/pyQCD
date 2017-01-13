cmake_minimum_required (VERSION 2.6)
project(pyQCD)

set (SRC
  bench_lattice.cpp
  bench_wilson_action.cpp)

foreach ( benchsourcefile ${SRC} )
  string( REPLACE ".cpp" "" benchname ${benchsourcefile} )
  get_filename_component( benchname ${benchname} NAME )
  add_executable( ${benchname} ${benchsourcefile} )
  target_link_libraries( ${benchname} pyQCDutils )
endforeach()