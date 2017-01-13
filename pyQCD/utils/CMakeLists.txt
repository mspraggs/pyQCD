cmake_minimum_required (VERSION 2.8)
project (pyQCD)

set (SRC
  math.cpp
  random.cpp)

add_library(pyQCDutils SHARED ${SRC})
target_link_libraries(pyQCDutils)