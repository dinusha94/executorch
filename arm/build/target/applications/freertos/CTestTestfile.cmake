# CMake generated Testfile for 
# Source directory: /home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u/core_platform/applications/freertos
# Build directory: /home/dinusha/simple_executorch/arm/build/target/applications/freertos
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[freertos]=] "/home/dinusha/miniconda3/envs/executorch/bin/python3.10" "/home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u/core_platform/targets/corstone-300/../../scripts/run_ctest.py" "-t" "corstone-300" "-a" "ethos-u55" "-m" "128" "/home/dinusha/simple_executorch/arm/build/target/applications/freertos/freertos.elf")
set_tests_properties([=[freertos]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u/core_platform/cmake/helpers.cmake;133;add_test;/home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u/core_platform/cmake/helpers.cmake;143;ethosu_add_test;/home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u/core_platform/applications/freertos/CMakeLists.txt;25;ethosu_add_executable_test;/home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u/core_platform/applications/freertos/CMakeLists.txt;0;")
