cmake_minimum_required(VERSION 3.14)
project(MyONNXApp)

set(CMAKE_CXX_STANDARD 17)

# ONNX Runtime
set(ORT_ROOT ${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime)
set(ORT_INCLUDE_DIR ${ORT_ROOT}/include/onnxruntime/core/session)
set(ORT_LIB_DIR ${ORT_ROOT}/build/MacOS/RelWithDebInfo)

# App sources
file(GLOB APP_SOURCES ${CMAKE_SOURCE_DIR}/app/*.cpp)

# Create executable from them
add_executable(my_app ${APP_SOURCES})

# Include ONNX Runtime headers
target_include_directories(my_app PRIVATE ${ORT_INCLUDE_DIR})

# Link ONNX Runtime library
target_link_directories(my_app PRIVATE ${ORT_LIB_DIR})
target_link_libraries(my_app PRIVATE onnxruntime)

message("Using ONNX Runtime from: ${ORT_ROOT}")
message("Include dir: ${ORT_INCLUDE_DIR}")
# Optional: set RPATH so you don’t need to export LD_LIBRARY_PATH
set_target_properties(my_app PROPERTIES
    INSTALL_RPATH ${ORT_LIB_DIR}
)
