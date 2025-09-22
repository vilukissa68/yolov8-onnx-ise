cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(ExternalProject)

# Use static libraries
set(BUILD_SHARED_LIBS OFF)
set(LIB_EXT ".a")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(OPENCV_LIB_DIR "lib64")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(OPENCV_LIB_DIR "lib")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(OPENCV_LIB_DIR "lib")
endif()


# OpenCV ExternalProject
set(OPENCV_PREFIX ${CMAKE_BINARY_DIR}/opencv)
set(OPENCV_INSTALL_DIR ${OPENCV_PREFIX}/install)

ExternalProject_Add(opencv_proj
    PREFIX ${OPENCV_PREFIX}
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv
    BINARY_DIR ${OPENCV_PREFIX}/build
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_LIST=core,imgproc,imgcodecs,dnn
        -DBUILD_ZLIB=OFF
        -DWITH_EIGEN=OFF
        -DWITH_IPP=OFF
        -DWITH_TBB=OFF
        -DWITH_CUDA=OFF
        -DWITH_OPENCL=OFF
        -DWITH_JPEG=ON
        -DWITH_PNG=OFF
        -DWITH_TIFF=OFF
        -DWITH_WEBP=OFF
        -DWITH_OPENEXR=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_DOCS=OFF
        -DWITH_IPP=OFF
        -DWITH_TBB=OFF
        -DWITH_OPENGL=OFF
        -DWITH_VTK=OFF
        -DWITH_FFMPEG=OFF
        -DWITH_GSTREAMER=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install -- -j${nproc}
    INSTALL_DIR ${OPENCV_INSTALL_DIR}
    BUILD_BYPRODUCTS
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_core${LIB_EXT}
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgproc${LIB_EXT}
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgcodecs${LIB_EXT}
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_dnn${LIB_EXT}
)

# Set OpenCV library paths
set(OpenCV_INCLUDE_DIRS ${OPENCV_INSTALL_DIR}/include/opencv4)
set(OPENCV_CORE_LIB ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_core${LIB_EXT})
set(OPENCV_IMGPROC_LIB ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgproc${LIB_EXT})
set(OPENCV_IMGCODECS_LIB ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgcodecs${LIB_EXT})
set(OPENCV_DNN_LIB ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_dnn${LIB_EXT})


# OpenCV dependnencies
set(OPENCV_DEP_LIB ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/opencv4/3rdparty)
set(OPENCV_LIB_NOTIF ${OPENCV_DEP_LIB}/libittnotify${LIB_EXT})
set(OPENCV_LIB_JPEG_TURBO ${OPENCV_DEP_LIB}/liblibjpeg-turbo${LIB_EXT})
set(OPENCV_LIB_OPEN_JP2 ${OPENCV_DEP_LIB}/liblibopenjp2${LIB_EXT})
set(OPENCV_LIB_PROTOBUF ${OPENCV_DEP_LIB}/liblibprotobuf${LIB_EXT})
set(OPENCV_LIB_HAL ${OPENCV_DEP_LIB}/libtegra_hal${LIB_EXT})

# ONNX Runtime ExternalProject
set(ORT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(ORT_BUILD_DIR ${ORT_SOURCE_DIR}/build/Linux/RelWithDebInfo)
  set(ORT_BUILD_COMMAND ./build.sh --config RelWithDebInfo --parallel --compile_no_warning_as_error --skip_submodule_sync)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(ORT_BUILD_DIR ${ORT_SOURCE_DIR}/build/MacOS/RelWithDebInfo)
  set(ORT_BUILD_COMMAND ./build.sh --config RelWithDebInfo --parallel --skip_tests --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64 )
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(ORT_BUILD_DIR ${ORT_SOURCE_DIR}/build/Windows/RelWithDebInfo)
  set(ORT_BUILD_COMMAND cmd /c build.bat --config RelWithDebInfo --parallel --compile_no_warning_as_error --skip_submodule_sync)
endif()

set(ORT_INCLUDE_DIR ${ORT_SOURCE_DIR}/include/onnxruntime/core/session)
set(ORT_LIB_COMMON ${ORT_BUILD_DIR}/libonnxruntime_common${LIB_EXT})
set(ORT_LIB_FLATBUFFERS ${ORT_BUILD_DIR}/libonnxruntime_flatbuffers${LIB_EXT})
set(ORT_LIB_FRAMEWORK ${ORT_BUILD_DIR}/libonnxruntime_framework${LIB_EXT})
set(ORT_LIB_GRAPH ${ORT_BUILD_DIR}/libonnxruntime_graph${LIB_EXT})
set(ORT_LIB_LORA ${ORT_BUILD_DIR}/libonnxruntime_lora${LIB_EXT})
set(ORT_LIB_MLAS ${ORT_BUILD_DIR}/libonnxruntime_mlas${LIB_EXT})
set(ORT_LIB_MOCKED_ALLOCATOR ${ORT_BUILD_DIR}/libonnxruntime_mocked_allocator${LIB_EXT})
set(ORT_LIB_OPTIMIZER ${ORT_BUILD_DIR}/libonnxruntime_optimizer${LIB_EXT})
set(ORT_LIB_PROVIDERS ${ORT_BUILD_DIR}/libonnxruntime_providers${LIB_EXT})
set(ORT_LIB_SESSION ${ORT_BUILD_DIR}/libonnxruntime_session${LIB_EXT})
set(ORT_LIB_TEST_UTILS ${ORT_BUILD_DIR}/libonnxruntime_test_utils${LIB_EXT})
set(ORT_LIB_UTIL ${ORT_BUILD_DIR}/libonnxruntime_util${LIB_EXT})


ExternalProject_Add(onnxruntime_proj
    PREFIX ${CMAKE_BINARY_DIR}/onnxruntime
    SOURCE_DIR ${ORT_SOURCE_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${ORT_BUILD_COMMAND}
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${ORT_LIB}
)

# Link dependencies and compile app
file(GLOB APP_SOURCES "app/*.cpp")
add_executable(my_app ${APP_SOURCES})

# Ensure app builds after dependencies
add_dependencies(my_app opencv_proj onnxruntime_proj)

# Include directories
target_include_directories(my_app PRIVATE
    ${ORT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(my_app PRIVATE
    ${OPENCV_CORE_LIB}
    ${OPENCV_IMGPROC_LIB}
    ${OPENCV_IMGCODECS_LIB}
    ${OPENCV_DNN_LIB}
    ${OPENCV_LIB_NOTIF}
    ${OPENCV_LIB_JPEG_TURBO}
    ${OPENCV_LIB_OPEN_JP2}
    ${OPENCV_LIB_PROTOBUF}
    ${OPENCV_LIB_HAL}
    ${ORT_LIB_COMMON}
    ${ORT_LIB_FLATBUFFERS}
    ${ORT_LIB_FRAMEWORK}
    ${ORT_LIB_GRAPH}
    ${ORT_LIB_LORA}
    ${ORT_LIB_MLAS}
    ${ORT_LIB_MOCKED_ALLOCATOR}
    ${ORT_LIB_OPTIMIZER}
    ${ORT_LIB_PROVIDERS}
    ${ORT_LIB_SESSION}
    ${ORT_LIB_TEST_UTILS}
    ${ORT_LIB_UTIL}
)
