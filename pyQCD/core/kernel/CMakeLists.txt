cmake_minimum_required (VERSION 2.8.8)
project (lattice)

option (USE_CUDA "Compile to use CUDA GPU?" 1)
option(USE_PYTHON3 "Compile for python 3?" 0)

set (BENCH_DIR ${PROJECT_BINARY_DIR}/benchmark)
set (SRC_DIR ${PROJECT_BINARY_DIR}/src)
set (INC_DIR ${PROJECT_BINARY_DIR}/include)
set (CUDA_INC_DIR ${PROJECT_BINARY_DIR}/cuda)

file (GLOB
  shared_SRC
  ${SRC_DIR}/lattice_*.cpp
  ${SRC_DIR}/random.cpp
  ${SRC_DIR}/utils.cpp
  ${SRC_DIR}/linear_operators/*.cpp
  ${SRC_DIR}/solvers.cpp)

file (GLOB
  lattice_SRC
  ${SRC_DIR}/pylattice.cpp
  ${SRC_DIR}/pyutils.cpp
  ${SRC_DIR}/wrapper.cpp)

file (GLOB
  cuda_SRC
  ${CUDA_INC_DIR}/*.cu)

file (GLOB
  benchmark_SRC
  RELATIVE
  ${PROJECT_BINARY_DIR}
  ${BENCH_DIR}/*.cpp)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set (EIGEN_INCLUDE_DIR ${CMAKE_MODULE_PATH}/FindEigen3.cmake)

if (USE_PYTHON3)
  message (STATUS "Compiling for Python 3")
  find_package (PythonLibs 3.0 REQUIRED)
else()
  message (STATUS "Compiling for Python 2")
  find_package (PythonLibs 2.6 REQUIRED)
endif()
find_package (Eigen3 3.1.3 REQUIRED)
find_package (Boost 1.46.0 REQUIRED)
find_package (OpenMP 3.0)
find_package (CUDA 4.2)

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
  ${INC_DIR}
  ${CUSP_INCLUDE_DIR}
  ${CUDA_INC_DIR})

set (Boost_USE_STATIC_LIBS OFF)
set (Boost_USE_MULTITHREADED ON)
set (Boost_USE_STATIC_RUNTIME OFF)
if (USE_PYTHON3)
  find_package (Boost COMPONENTS random python3 timer system)
  if (NOT Boost_FOUND)
    find_package (Boost COMPONENTS random python-py34 timer system)
  endif()
  if (NOT Boost_FOUND)
    find_package (Boost COMPONENTS random python-py33 timer system)
  endif()
  if (NOT Boost_FOUND)
    find_package (Boost COMPONENTS random python-py32 timer system)
  endif()
  if (NOT Boost_FOUND)
    find_package (Boost COMPONENTS random python-py31 timer system)
  endif()
  if (NOT Boost_FOUND)
    find_package (Boost COMPONENTS random python-py30 timer system)
  endif()
else()
  find_package (Boost COMPONENTS random python timer system)
endif()


#set (CMAKE_SHARED_LIBRARY_PREFIX "")

if (CUDA_FOUND AND USE_CUDA)
  message (STATUS "Compiling for CUDA GPUs")

  if (NOT CUSP_INCLUDE_DIR)
    message (STATUS "Warning! Path to CUSP not defined!")
    message (STATUS "Falling back to CPU sparse solvers.")
  else()
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    add_definitions(-DUSE_CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler '-fPIC';-O3)
    cuda_add_library(cudalinops ${cuda_SRC})
    link_directories(${PROJECT_BINARY_DIR})
  endif()
endif()

add_library(shared_objects OBJECT ${shared_SRC})

add_library(pyQCD SHARED $<TARGET_OBJECTS:shared_objects>)
target_link_libraries(pyQCD ${Boost_LIBRARIES})
add_library(pyQCDstatic STATIC $<TARGET_OBJECTS:shared_objects>)
target_link_libraries(pyQCDstatic ${Boost_LIBRARIES})

add_library (lattice SHARED $<TARGET_OBJECTS:shared_objects> ${lattice_SRC})
set_target_properties(lattice PROPERTIES PREFIX "")

if (CUDA_FOUND AND USE_CUDA AND CUSP_INCLUDE_DIR)
  target_link_libraries (lattice ${Boost_LIBRARIES} ${PYTHON_LIBRARIES}
    cudalinops ${CUDA_LIBRARIES})
else()
  target_link_libraries (lattice ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
endif()

foreach ( benchsourcefile ${benchmark_SRC} )
  string( REPLACE ".cpp" "" benchname ${benchsourcefile} )
  add_executable( ${benchname} $<TARGET_OBJECTS:shared_objects>
    ${benchsourcefile} )
  if (CUDA_FOUND AND USE_CUDA AND CUSP_INCLUDE_DIR)
    target_link_libraries( ${benchname} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES}
      cudalinops ${CUDA_LIBRARIES})
  else()
    target_link_libraries( ${benchname} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
  endif()
endforeach ( benchsourcefile ${benchmark_SRC} )

IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_DEFINITIONS("-O3 -ffast-math")
ENDIF()
IF(CMAKE_COMPILER_IS_MSVC)
  ADD_DEFINITIONS("/O2")
ENDIF()
IF(UNIX AND NOT WIN32)
  FIND_PROGRAM(CMAKE_UNAME uname /bin /usr/bin /usr/local/bin )
  IF(CMAKE_UNAME)
    EXEC_PROGRAM(uname ARGS -m OUTPUT_VARIABLE CMAKE_SYSTEM_PROCESSOR)
    SET(CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR} CACHE INTERNAL
"processor type (i386 and x86_64)")
    IF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      ADD_DEFINITIONS(-fPIC)
    ENDIF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  ENDIF(CMAKE_UNAME)
ENDIF(UNIX AND NOT WIN32)
