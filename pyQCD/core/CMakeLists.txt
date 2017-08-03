cmake_minimum_required (VERSION 2.8)
project (pyQCD)

set (SRC
  layout.cpp)

add_library(pyQCDcore SHARED ${SRC})
target_link_libraries(pyQCDcore)