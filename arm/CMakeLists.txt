
# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)
project(arm_simple_app)


option(ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE "Set ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE to specify memory alloction pool size" OFF)
option(ET_ARM_BAREMETAL_TEMP_ALLOCATOR_POOL_SIZE "Set ET_ARM_BAREMETAL_TEMP_ALLOCATOR_POOL_SIZE to specify temp alloction pool size" OFF)


# Selects timing adapter values matching system_config.
# Default is Ethos_U55_High_End_Embedded, simulating optimal hardware for the Corestone-300.
set(SYSTEM_CONFIG "Ethos_U55_High_End_Embedded" CACHE STRING "System config")

# Dependencies from the Ethos-U Core This is the platform target of
# Corstone-300, that includes ethosu_core_driver and bare-metal bringup
# libraries. We link against ethosu_target_init which includes all of these
# dependencies.
# if(SYSTEM_CONFIG STREQUAL "Ethos_U55_High_End_Embedded")
  add_subdirectory(/home/dinusha/executorch/examples/arm/ethos-u-scratch/ethos-u/core_platform/targets/corstone-300 target)
  set(TARGET_BOARD "corstone-300")
  target_compile_definitions(ethosu_target_common INTERFACE
      # ETHOSU_MODEL=0 place pte file/data in SRAM area
      # ETHOSU_MODEL=1 place pte file/data in DDR area
      ETHOSU_MODEL=1
      # Configure NPU architecture timing adapters
      # This is just example numbers and you should make this match your hardware
      # SRAM
      ETHOSU_TA_MAXR_0=8
      ETHOSU_TA_MAXW_0=8
      ETHOSU_TA_MAXRW_0=0
      ETHOSU_TA_RLATENCY_0=32
      ETHOSU_TA_WLATENCY_0=32
      ETHOSU_TA_PULSE_ON_0=3999
      ETHOSU_TA_PULSE_OFF_0=1
      ETHOSU_TA_BWCAP_0=4000
      ETHOSU_TA_PERFCTRL_0=0
      ETHOSU_TA_PERFCNT_0=0
      ETHOSU_TA_MODE_0=1
      ETHOSU_TA_HISTBIN_0=0
      ETHOSU_TA_HISTCNT_0=0
      # Flash
      ETHOSU_TA_MAXR_1=2
      ETHOSU_TA_MAXW_1=0
      ETHOSU_TA_MAXRW_1=0
      ETHOSU_TA_RLATENCY_1=64
      ETHOSU_TA_WLATENCY_1=0
      ETHOSU_TA_PULSE_ON_1=320
      ETHOSU_TA_PULSE_OFF_1=80
      ETHOSU_TA_BWCAP_1=50
      ETHOSU_TA_PERFCTRL_1=0
      ETHOSU_TA_PERFCNT_1=0
      ETHOSU_TA_MODE_1=1
      ETHOSU_TA_HISTBIN_1=0
      ETHOSU_TA_HISTCNT_1=0
      )


# Dependencies from the ExecuTorch build
add_library(executorch STATIC IMPORTED)
set_property(
  TARGET executorch PROPERTY IMPORTED_LOCATION
                             "/usr/local/lib/arm/libexecutorch.a"
)

add_library(executorch_core STATIC IMPORTED)
set_property(
  TARGET executorch_core
  PROPERTY IMPORTED_LOCATION "/usr/local/lib/arm/libexecutorch_core.a"
)
target_link_libraries(executorch INTERFACE executorch_core)

add_library(executorch_delegate_ethos_u STATIC IMPORTED)
set_property(
  TARGET executorch_delegate_ethos_u
  PROPERTY IMPORTED_LOCATION
           "/usr/local/lib/arm/libexecutorch_delegate_ethos_u.a"
)

add_library(portable_ops_lib STATIC IMPORTED)
set_property(
  TARGET portable_ops_lib
  PROPERTY IMPORTED_LOCATION
           "/usr/local/lib/arm/libarm_portable_ops_lib.a"
)
add_library(portable_kernels STATIC IMPORTED)
set_property(
  TARGET portable_kernels
  PROPERTY IMPORTED_LOCATION
           "/usr/local/lib/arm/libportable_kernels.a"
)

add_library(quantized_ops_lib STATIC IMPORTED)
set_property(
  TARGET quantized_ops_lib
  PROPERTY IMPORTED_LOCATION
           "/usr/local/lib/arm/libquantized_ops_lib.a"
)
add_library(quantized_kernels STATIC IMPORTED)
set_property(
  TARGET quantized_kernels
  PROPERTY IMPORTED_LOCATION
           "/usr/local/lib/arm/libquantized_kernels.a"
)

add_library(extension_runner_util STATIC IMPORTED)
set_property(
  TARGET extension_runner_util
  PROPERTY
    IMPORTED_LOCATION
    "/usr/local/lib/arm/libextension_runner_util.a"
)


# The arm_simple_app executable
add_executable(arm_simple_app)

target_sources(
  arm_simple_app PRIVATE 
  src/main.cpp
)

# Include the target's bare-metal linker script
ethosu_eval_link_options(arm_simple_app)

set(arm_executor_runner_link)
list(APPEND arm_executor_runner_link
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
  -Xlinker -Map=arm_simple_app.map
)

# Need whole-archive to ensure C++ ctor's are called - this may be wasteful for
# bin size as we link in a number of other symbols
target_link_libraries(
  arm_simple_app
  ${arm_executor_runner_link}
)

target_link_options( arm_simple_app PUBLIC LINKER:-Map=arm_simple_app.map )

# ET headers and generated headers includes
target_include_directories(
  arm_simple_app PRIVATE 
  /usr/local/include/ 
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/include/

)


if(ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(arm_simple_app PUBLIC ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE})
endif()

if(ET_ARM_BAREMETAL_TEMP_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(arm_simple_app PUBLIC ET_ARM_BAREMETAL_TEMP_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_TEMP_ALLOCATOR_POOL_SIZE})
endif()


