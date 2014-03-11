cmake_minimum_required (VERSION 2.8.8)
project (lattice)
enable_testing()

option (USE_CUDA "Compile to use CUDA GPU?" 1)

set (TEST_DIR ${PROJECT_BINARY_DIR}/test)
set (SRC_DIR ${PROJECT_BINARY_DIR}/src)
set (INC_DIR ${PROJECT_BINARY_DIR}/include)

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
  test_SRC
  ${TEST_DIR}/*.cpp)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set (EIGEN_INCLUDE_DIR ${CMAKE_MODULE_PATH}/FindEigen3.cmake)

find_package (PythonLibs 2.6 REQUIRED)
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
  ${CUSP_INCLUDE_DIR})

set (Boost_USE_STATIC_LIBS OFF)
set (Boost_USE_MULTITHREADED ON)
set (Boost_USE_STATIC_RUNTIME OFF)
find_package (Boost COMPONENTS random python unit_test_framework)

set (CMAKE_SHARED_LIBRARY_PREFIX "")

if (CUDA_FOUND AND USE_CUDA)
  message ("Compiling for CUDA GPUs")

  if (NOT CUSP_INCLUDE_DIR)
    message ("Error! Path to CUSP not defined!")
    message ("Falling back to Eigen sparse solvers.")
  else()
    add_definitions(-DUSE_CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler '-fPIC';-O3)
    cuda_add_library(cudainverters src/cuda_utils.cu)
    link_directories(${PROJECT_BINARY_DIR})
  endif()
endif()

add_library(shared_objects OBJECT ${shared_SRC})

add_library (lattice SHARED $<TARGET_OBJECTS:shared_objects> ${lattice_SRC})

if (CUDA_FOUND AND USE_CUDA AND CUSP_INCLUDE_DIR)
  target_link_libraries (lattice ${Boost_LIBRARIES} ${PYTHON_LIBRARIES}
    cudainverters ${CUDA_LIBRARIES})
else()
  target_link_libraries (lattice ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
endif()

add_definitions(-DBOOST_TEST_DYN_LINK)
add_executable (test_lattice $<TARGET_OBJECTS:shared_objects> ${test_SRC})

if (CUDA_FOUND AND USE_CUDA AND CUSP_INCLUDE_DIR)
  target_link_libraries (test_lattice ${Boost_LIBRARIES} ${PYTHON_LIBRARIES}
    cudainverters ${CUDA_LIBRARIES})
else()
  target_link_libraries (test_lattice ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
endif()

add_test(lattice test_lattice)

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
