cmake_minimum_required(VERSION 3.19)
project(SimpleProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add the main executable
add_executable(SimpleApp src/main.cpp)

# Link against the ExecuTorch library and additional dependencies
target_link_libraries(SimpleApp PRIVATE
    /usr/local/lib/libexecutorch.a
    /usr/local/lib/libextension_module_static.a
    /usr/local/lib/libexecutorch_no_prim_ops.a
    /usr/local/lib/libextension_data_loader.a
    /usr/local/lib/libextension_tensor.a
    -Wl,--whole-archive /usr/local/lib/libportable_ops_lib.a -Wl,--no-whole-archive
    /usr/local/lib/libportable_kernels.a

)

# Include directories for ExecuTorch
target_include_directories(SimpleApp PRIVATE
    /usr/local/include/executorch
)
