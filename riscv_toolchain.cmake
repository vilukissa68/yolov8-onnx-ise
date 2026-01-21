# riscv_toolchain.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Define Compilers
set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)

# Define Sysroot
set(CMAKE_SYSROOT ${HOME}/riscv64-sysroot)

# Setup Search Modes
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Skip tests
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(OpenCL_INCLUDE_DIR ${CMAKE_SYSROOT}/usr/include)

# Try to find the generic ICD loader in standard locations inside sysroot
file(GLOB OpenCL_LIBRARY_PATH 
    "${CMAKE_SYSROOT}/usr/lib/riscv64-linux-gnu/libOpenCL.so*"
    "${CMAKE_SYSROOT}/usr/lib/libOpenCL.so*"
)

if(OpenCL_LIBRARY_PATH)
    set(OpenCL_LIBRARY ${OpenCL_LIBRARY_PATH})
else()
    message(WARNING "Could not auto-locate libOpenCL.so in Sysroot. Compilation might fail.")
endif()
