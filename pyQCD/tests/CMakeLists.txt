cmake_minimum_required (VERSION 2.8)
project(pyQCD)

set (SRC
  test_gamma_matrices.cpp
  test_heatbath.cpp
  test_lattice.cpp
  test_layout.cpp
  test_main.cpp
  test_random.cpp
  test_wilson_gauge_action.cpp
  test_rectangle_gauge_action.cpp
  test_wilson_heatbath_update.cpp
  test_rectangle_heatbath_update.cpp
  test_hopping_matrix.cpp
  test_wilson_fermion_action.cpp
  test_conjugate_gradient.cpp)

add_executable(run_tests ${SRC})
target_link_libraries(run_tests pyQCDutils)
add_test(run_tests --log_level=all)