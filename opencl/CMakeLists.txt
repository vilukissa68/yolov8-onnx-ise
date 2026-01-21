cmake_minimum_required(VERSION 3.14)
project(yolo_opencl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(ExternalProject)

# Setup TVM_HOME
if(NOT DEFINED TVM_HOME)
    set(TVM_HOME $ENV{TVM_HOME})
endif()

if(NOT EXISTS "${TVM_HOME}")
    message(FATAL_ERROR "TVM_HOME not found. Please set export TVM_HOME=/path/to/tvm or run cmake -DTVM_HOME=...")
endif()

message(STATUS "Using TVM_HOME: ${TVM_HOME}")

# Define TVM includes
set(TVM_INCLUDES
    ${TVM_HOME}/include
    ${TVM_HOME}/3rdparty/dlpack/include
    ${TVM_HOME}/3rdparty/dmlc-core/include
)

# Define TVM Runtime Library
set(TVM_RUNTIME_LIB "${TVM_HOME}/build/libtvm_runtime.so")

if(NOT EXISTS "${TVM_RUNTIME_LIB}")
    message(FATAL_ERROR "libtvm_runtime.so not found at ${TVM_RUNTIME_LIB}. Did you build TVM?")
endif()

# Setup OpenCL
find_package(OpenCL REQUIRED)

# Deploy directory for .so/.dll
set(DEPLOY_DIR ${CMAKE_BINARY_DIR}/libs)
file(MAKE_DIRECTORY ${DEPLOY_DIR})

# Detect platform shared library extension
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LIB_EXT ".so")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(LIB_EXT ".dylib")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(LIB_EXT ".dll")
endif()

set(OPENCV_LIB_DIR "lib")

# Build OpenCV as ExternalProject
set(OPENCV_PREFIX ${CMAKE_BINARY_DIR}/opencv)
set(OPENCV_INSTALL_DIR ${OPENCV_PREFIX}/install)

ExternalProject_Add(opencv_proj
    PREFIX ${OPENCV_PREFIX}
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv
    BINARY_DIR ${OPENCV_PREFIX}/build
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
        # --- FIX 2: FORCE INSTALL LOCATION ---
        -DCMAKE_INSTALL_LIBDIR=${OPENCV_LIB_DIR} 
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_LIST=core,imgproc,imgcodecs,dnn
        -DBUILD_ZLIB=OFF
        -DWITH_EIGEN=OFF
        -DWITH_IPP=OFF
        -DWITH_TBB=OFF
        -DWITH_CUDA=OFF
        -DWITH_OPENCL=OFF
        -DBUILD_TESTS=OFF           
        -DBUILD_PERF_TESTS=OFF      
        -DBUILD_EXAMPLES=OFF        
        -DBUILD_opencv_apps=OFF     
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install -- -j8
    INSTALL_DIR ${OPENCV_INSTALL_DIR}
    BUILD_BYPRODUCTS
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_core${LIB_EXT}
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgproc${LIB_EXT}
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgcodecs${LIB_EXT}
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_dnn${LIB_EXT}
)

# Set OpenCV library paths (Now guaranteed to be in 'lib')
set(OpenCV_INCLUDE_DIRS ${OPENCV_INSTALL_DIR}/include/opencv4)
set(OPENCV_CORE_LIB ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_core${LIB_EXT})
set(OPENCV_IMGPROC_LIB ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgproc${LIB_EXT})
set(OPENCV_IMGCODECS_LIB ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgcodecs${LIB_EXT})
set(OPENCV_DNN_LIB ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_dnn${LIB_EXT})

ExternalProject_Add_Step(opencv_proj copy_libs
    DEPENDEES install   # Wait for install, not just build
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENCV_CORE_LIB} ${DEPLOY_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENCV_IMGPROC_LIB} ${DEPLOY_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENCV_IMGCODECS_LIB} ${DEPLOY_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENCV_DNN_LIB} ${DEPLOY_DIR}/
    COMMENT "Copying OpenCV libraries to ${DEPLOY_DIR}"
)

# Link dependencies and compile app
file(GLOB APP_SOURCES "*.cpp")
add_executable(yolo_opencl ${APP_SOURCES})

# Ensure app builds after dependencies
add_dependencies(yolo_opencl opencv_proj)

# Include directories
target_include_directories(yolo_opencl PRIVATE
    ${TVM_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
# Note: We link against the installed versions
target_link_libraries(yolo_opencl PRIVATE
    ${OPENCV_CORE_LIB}
    ${OPENCV_IMGPROC_LIB}
    ${OPENCV_IMGCODECS_LIB}
    ${OPENCV_DNN_LIB}
	OpenCL::OpenCL
	${TVM_RUNTIME_LIB}
)

# This tells the executable to look in the TVM build folder for the runtime .so
set_target_properties(yolo_opencl PROPERTIES
    BUILD_RPATH "${TVM_HOME}/build"
    INSTALL_RPATH "${TVM_HOME}/build"
)

set(MODEL_GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/generate_opencl_model.py")
set(MODEL_LIB_NAME "yolov8_opencl.so")
set(PYTHON_OUTPUT_DIR "${CMAKE_BINARY_DIR}/tvm_c_out")
set(GENERATED_MODEL_PATH "${PYTHON_OUTPUT_DIR}/${MODEL_LIB_NAME}")

file(MAKE_DIRECTORY ${PYTHON_OUTPUT_DIR})

add_custom_command(
    OUTPUT ${GENERATED_MODEL_PATH}
    ${Python3_EXECUTABLE}
    COMMAND ${Python3_EXECUTABLE} ${MODEL_GEN_SCRIPT} --output ${PYTHON_OUTPUT_DIR}
    DEPENDS ${MODEL_GEN_SCRIPT}
    COMMENT "Running Python script to generate TVM Model Library..."
)

add_custom_target(generate_model DEPENDS ${GENERATED_MODEL_PATH})

add_dependencies(yolo_opencl generate_model)

add_custom_command(TARGET yolo_opencl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${GENERATED_MODEL_PATH}
    $<TARGET_FILE_DIR:yolo_opencl>/${MODEL_LIB_NAME}
    COMMENT "Copying ${MODEL_LIB_NAME} to binary directory..."
)
