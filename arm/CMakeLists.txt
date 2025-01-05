# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)
project(arm_simple_app)

set(ETHOS_SDK_PATH
    "/home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u"
    CACHE PATH "Path to Ethos-U bare metal driver/env"
)



# Dependencies from the Ethos-U Core This is the platform target of
# Corstone-300, that includes ethosu_core_driver and bare-metal bringup
# libraries. We link against ethosu_target_init which includes all of these
# dependencies.

# For Corstone-300 FVP builds we put models into the larger DRAM area
set(MEMORY_MODEL "dram")
set(MEMORY_ARENA "dram")
add_subdirectory(${ETHOS_SDK_PATH}/core_platform/targets/corstone-300 target)


# Dependencies from the ExecuTorch build
add_library(executorch STATIC IMPORTED)
set_property(
  TARGET executorch PROPERTY IMPORTED_LOCATION
                            #  "${ET_BUILD_DIR_PATH}/libexecutorch.a"
                            "/usr/local/lib/arm/libexecutorch.a"
)

add_library(executorch_no_prim_ops STATIC IMPORTED)
set_property(
  TARGET executorch_no_prim_ops
  PROPERTY IMPORTED_LOCATION 
                            # "${ET_BUILD_DIR_PATH}/libexecutorch_no_prim_ops.a"
                            "/usr/local/lib/arm/libexecutorch_no_prim_ops.a"
)
target_link_libraries(executorch INTERFACE executorch_no_prim_ops)

add_library(executorch_delegate_ethos_u STATIC IMPORTED)
set_property(
  TARGET executorch_delegate_ethos_u
  PROPERTY IMPORTED_LOCATION
          #  "${ET_BUILD_DIR_PATH}/backends/arm/libexecutorch_delegate_ethos_u.a"
          "/usr/local/lib/arm/libexecutorch_delegate_ethos_u.a"
)

add_library(portable_ops_lib STATIC IMPORTED)
set_property(
  TARGET portable_ops_lib
  PROPERTY IMPORTED_LOCATION
          #  "${ET_BUILD_DIR_PATH}/examples/arm/libarm_portable_ops_lib.a"
          "/usr/local/lib/arm/libarm_portable_ops_lib.a"
)
add_library(portable_kernels STATIC IMPORTED)
set_property(
  TARGET portable_kernels
  PROPERTY IMPORTED_LOCATION
          #  "${ET_BUILD_DIR_PATH}/kernels/portable/libportable_kernels.a"
          "/usr/local/lib/arm/libportable_kernels.a"
)

add_library(quantized_ops_lib STATIC IMPORTED)
set_property(
  TARGET quantized_ops_lib
  PROPERTY IMPORTED_LOCATION
          #  "${ET_BUILD_DIR_PATH}/kernels/quantized/libquantized_ops_lib.a"
          "/usr/local/lib/arm/libquantized_ops_lib.a"
)
add_library(quantized_kernels STATIC IMPORTED)
set_property(
  TARGET quantized_kernels
  PROPERTY IMPORTED_LOCATION
          #  "${ET_BUILD_DIR_PATH}/kernels/quantized/libquantized_kernels.a"
          "/usr/local/lib/arm/libquantized_kernels.a"
)

add_library(extension_runner_util STATIC IMPORTED)
set_property(
  TARGET extension_runner_util
  PROPERTY
    IMPORTED_LOCATION
    # "${ET_BUILD_DIR_PATH}/extension/runner_util/libextension_runner_util.a"
    "/usr/local/lib/arm/libextension_runner_util.a"
)


# The arm_executor_runner executable
add_executable(arm_simple_app)

target_sources(
  arm_simple_app PRIVATE src/main.cpp #arm_perf_monitor.cpp
)

# Include the target's bare-metal linker script
ethosu_eval_link_options(arm_simple_app)

# Need whole-archive to ensure C++ ctor's are called - this may be wasteful for
# bin size as we link in a number of other symbols
target_link_libraries(
  arm_simple_app
  extension_runner_util
  ethosu_target_init
  executorch
  "-Wl,--whole-archive"
  executorch_delegate_ethos_u
  quantized_ops_lib
  portable_ops_lib
  quantized_kernels
  portable_kernels
  "-Wl,--no-whole-archive"
)

message(STATUS "current bin dir '${CMAKE_CURRENT_BINARY_DIR}'")

# ET headers and generated headers includes
target_include_directories(
  arm_simple_app PRIVATE /usr/local/include/ 
  ${CMAKE_CURRENT_BINARY_DIR}
)


