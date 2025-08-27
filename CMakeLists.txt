cmake_minimum_required(VERSION 3.14)
project(MyONNXApp)

set(CMAKE_CXX_STANDARD 17)

# Set ONNX Runtime paths relative to this file
set(ORT_ROOT ${CMAKE_SOURCE_DIR}/../3rdparty/onnxruntime)
set(ORT_INCLUDE_DIR ${ORT_ROOT}/include)
set(ORT_LIB_DIR ${ORT_ROOT}/build/Linux/Release/lib)

# Your source files
add_executable(my_app main.cpp)

# Include ONNX Runtime headers
target_include_directories(my_app PRIVATE ${ORT_INCLUDE_DIR})

# Link ONNX Runtime library
target_link_directories(my_app PRIVATE ${ORT_LIB_DIR})
target_link_libraries(my_app PRIVATE onnxruntime)

# Optional: set RPATH so you don’t need to export LD_LIBRARY_PATH
set_target_properties(my_app PROPERTIES
    INSTALL_RPATH ${ORT_LIB_DIR}
)
