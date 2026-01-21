cmake_minimum_required(VERSION 3.14)
project(yolo_opencl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(ExternalProject)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(OpenCL REQUIRED)

option(USE_QUANTIZATION "Enable Int8 Quantization for the TVM Model" OFF)

if(USE_QUANTIZATION)
    message(STATUS "Quantization: ENABLED (Int8)")
    set(QUANTIZE_FLAG "--quantize")
else()
    message(STATUS "Quantization: DISABLED (FP32)")
    set(QUANTIZE_FLAG "")
endif()

# Check that TVM Python bindgins are available
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import tvm; print(tvm.__version__)"
    RESULT_VARIABLE TVM_PY_CHECK
    OUTPUT_QUIET
    ERROR_QUIET
)

if(NOT TVM_PY_CHECK EQUAL 0)
    message(WARNING 
        "----------------------------------------------------------------\n"
        " Python package 'apache-tvm' not found in ${Python3_EXECUTABLE}.\n"
        " The model generation step will fail unless you install it:\n"
        "    pip install apache-tvm\n"
        "----------------------------------------------------------------"
    )
endif()

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)
message(STATUS "Project Root Directory: ${PROJECT_ROOT_DIR}")

# TVM runtime build
set(TVM_PREFIX ${CMAKE_BINARY_DIR}/tvm_build)
set(TVM_INSTALL_DIR ${TVM_PREFIX}/install)
set(TVM_SOURCE_DIR ${PROJECT_ROOT_DIR}/3rdparty/tvm) # Path to submodule

if(NOT EXISTS "${TVM_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "TVM submodule missing. Run: git submodule update --init --recursive")
endif()

# Runtime-only configuration (Fast build, no LLVM)
set(TVM_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${TVM_INSTALL_DIR}
    -DCMAKE_BUILD_TYPE=Release
    -DUSE_OPENCL=ON
    -DUSE_LLVM=OFF
    -DUSE_MICRO=OFF
    -DUSE_RPC=OFF
    -DUSE_CPP_RPC=OFF
    -DUSE_GRAPH_EXECUTOR=ON
    -DUSE_PROFILER=OFF
)

# Forward Cross-Compilation Toolchain if present
if(CMAKE_TOOLCHAIN_FILE)
    list(APPEND TVM_CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
endif()

ExternalProject_Add(tvm_runtime_proj
    PREFIX ${TVM_PREFIX}
    SOURCE_DIR ${TVM_SOURCE_DIR}
    CMAKE_ARGS ${TVM_CMAKE_ARGS}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel 10
)

# Define Artifact Paths
set(TVM_INCLUDES
    ${TVM_INSTALL_DIR}/include
    ${TVM_SOURCE_DIR}/3rdparty/dlpack/include
    ${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include
)

if(APPLE)
    set(LIB_EXT ".dylib")
else()
    set(LIB_EXT ".so")
endif()

set(TVM_RUNTIME_LIB "${TVM_INSTALL_DIR}/lib/libtvm_runtime${LIB_EXT}")

# Build OpenCV
set(DEPLOY_DIR ${CMAKE_BINARY_DIR}/libs)
file(MAKE_DIRECTORY ${DEPLOY_DIR})

set(OPENCV_LIB_DIR "lib")
set(OPENCV_PREFIX ${CMAKE_BINARY_DIR}/opencv)
set(OPENCV_INSTALL_DIR ${OPENCV_PREFIX}/install)

ExternalProject_Add(opencv_proj
    PREFIX ${OPENCV_PREFIX}
    SOURCE_DIR ${PROJECT_ROOT_DIR}/3rdparty/opencv
    BINARY_DIR ${OPENCV_PREFIX}/build
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
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
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install -- -j${nproc}
    INSTALL_DIR ${OPENCV_INSTALL_DIR}
    BUILD_BYPRODUCTS
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_core${LIB_EXT}
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgproc${LIB_EXT}
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgcodecs${LIB_EXT}
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_dnn${LIB_EXT}
)

set(OpenCV_INCLUDE_DIRS ${OPENCV_INSTALL_DIR}/include/opencv4)
set(OPENCV_LIBS
    ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_core${LIB_EXT}
    ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgproc${LIB_EXT}
    ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgcodecs${LIB_EXT}
    ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_dnn${LIB_EXT}
)

ExternalProject_Add_Step(opencv_proj copy_libs
    DEPENDEES install
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENCV_LIBS} ${DEPLOY_DIR}/
)

# Generate TVM Model using Python Script
set(MODEL_GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/generate_opencl_model.py")
set(PYTHON_OUTPUT_DIR "${CMAKE_BINARY_DIR}/tvm_c_out")

file(MAKE_DIRECTORY ${PYTHON_OUTPUT_DIR})

# Choose target architecture
if(CMAKE_CROSSCOMPILING AND CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
    message(STATUS "Cross-compiling detected: Generating RISC-V Model...")
    set(TARGET_ARCH_FLAG "riscv64")
    set(MODEL_LIB_NAME "yolov8_opencl_riscv.so")
else()
    message(STATUS "Native compilation: Generating Host Model...")
    set(TARGET_ARCH_FLAG "native")
    set(MODEL_LIB_NAME "yolov8_opencl.so")
endif()

set(GENERATED_MODEL_PATH "${PYTHON_OUTPUT_DIR}/${MODEL_LIB_NAME}")

add_custom_command(
    OUTPUT ${GENERATED_MODEL_PATH}
    COMMAND ${Python3_EXECUTABLE} ${MODEL_GEN_SCRIPT} --output ${PYTHON_OUTPUT_DIR} --target-arch ${TARGET_ARCH_FLAG} --quantize ${QUANTIZE_FLAG}
    DEPENDS ${MODEL_GEN_SCRIPT}
    COMMENT "Generating TVM Model using Host Python (requires 'pip install apache-tvm')..."
)

add_custom_target(generate_model DEPENDS ${GENERATED_MODEL_PATH})

# Compile application
file(GLOB APP_SOURCES "*.cpp")
add_executable(yolo_opencl ${APP_SOURCES})

add_dependencies(yolo_opencl generate_model opencv_proj tvm_runtime_proj)

target_include_directories(yolo_opencl PRIVATE
    ${TVM_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo_opencl PRIVATE
    ${OPENCV_LIBS}
    ${TVM_RUNTIME_LIB}
    OpenCL::OpenCL
)

# Copy Runtime to libs/
add_custom_command(TARGET yolo_opencl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TVM_RUNTIME_LIB}
    ${DEPLOY_DIR}/libtvm_runtime${LIB_EXT}
    COMMENT "Copying TVM Runtime to libs dir..."
)

# Copy Model to bin/
add_custom_command(TARGET yolo_opencl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${GENERATED_MODEL_PATH}
    $<TARGET_FILE_DIR:yolo_opencl>/${MODEL_LIB_NAME}
)

# RPATH Setup (Look in ./libs)
set_target_properties(yolo_opencl PROPERTIES
    BUILD_RPATH "${DEPLOY_DIR}"
    INSTALL_RPATH "${DEPLOY_DIR}"
    LINK_FLAGS "-Wl,-rpath,${DEPLOY_DIR}"
)
