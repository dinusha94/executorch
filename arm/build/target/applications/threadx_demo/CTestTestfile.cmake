# CMake generated Testfile for 
# Source directory: /home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u/core_platform/applications/threadx_demo
# Build directory: /home/dinusha/simple_executorch/arm/build/target/applications/threadx_demo
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[threadx_demo]=] "/home/dinusha/miniconda3/envs/executorch/bin/python3.10" "/home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u/core_platform/targets/corstone-300/../../scripts/run_ctest.py" "-t" "corstone-300" "-a" "ethos-u55" "-m" "128" "/home/dinusha/simple_executorch/arm/build/target/applications/threadx_demo/threadx_demo.elf")
set_tests_properties([=[threadx_demo]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u/core_platform/cmake/helpers.cmake;133;add_test;/home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u/core_platform/cmake/helpers.cmake;143;ethosu_add_test;/home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u/core_platform/applications/threadx_demo/CMakeLists.txt;25;ethosu_add_executable_test;/home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u/core_platform/applications/threadx_demo/CMakeLists.txt;0;")
