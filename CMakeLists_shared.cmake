cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(ExternalProject)

# Detect platform shared library extension
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LIB_EXT ".so")
    set(OPENCV_LIB_DIR "lib64")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(LIB_EXT ".dylib")
    set(OPENCV_LIB_DIR "lib")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(LIB_EXT ".dll")
    set(OPENCV_LIB_DIR "lib")
endif()

# Deploy directory for .so/.dll
set(DEPLOY_DIR ${CMAKE_BINARY_DIR}/libs)
file(MAKE_DIRECTORY ${DEPLOY_DIR})

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
        -DBUILD_LIST=core,imgproc,imgcodecs,dnn
        -DBUILD_ZLIB=OFF
        -DWITH_EIGEN=OFF
        -DWITH_IPP=OFF
        -DWITH_TBB=OFF
        -DWITH_CUDA=OFF
        -DWITH_OPENCL=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install -- -j${nproc}
    INSTALL_DIR ${OPENCV_INSTALL_DIR}
    BUILD_BYPRODUCTS
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_core${LIB_EXT}
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgproc${LIB_EXT}
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgcodecs${LIB_EXT}
        ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_dnn${LIB_EXT}
        #${OPENCV_INSTALL_DIR}/lib64/libopencv_highgui${LIB_EXT}
)

# Set OpenCV library paths
set(OpenCV_INCLUDE_DIRS ${OPENCV_INSTALL_DIR}/include/opencv4)
set(OPENCV_CORE_LIB ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_core${LIB_EXT})
set(OPENCV_IMGPROC_LIB ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgproc${LIB_EXT})
set(OPENCV_IMGCODECS_LIB ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_imgcodecs${LIB_EXT})
set(OPENCV_DNN_LIB ${OPENCV_INSTALL_DIR}/${OPENCV_LIB_DIR}/libopencv_dnn${LIB_EXT})
#set(OPENCV_HIGHGUI_LIB ${OPENCV_INSTALL_DIR}/lib64/libopencv_highgui${LIB_EXT})

# Copy OpenCV .so to deploy dir
ExternalProject_Add_Step(opencv_proj copy_libs
    DEPENDEES build
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENCV_CORE_LIB} ${DEPLOY_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENCV_IMGPROC_LIB} ${DEPLOY_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENCV_IMGCODECS_LIB} ${DEPLOY_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENCV_DNN_LIB} ${DEPLOY_DIR}/
    #COMMAND ${CMAKE_COMMAND} -E copy ${OPENCV_HIGHGUI_LIB} ${DEPLOY_DIR}/
    COMMENT "Copy OpenCV shared libraries to deploy folder"
)

# ONNX Runtime ExternalProject
set(ORT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(ORT_BUILD_DIR ${ORT_SOURCE_DIR}/build/Linux/RelWithDebInfo)
  set(ORT_BUILD_COMMAND ./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(ORT_BUILD_DIR ${ORT_SOURCE_DIR}/build/MacOS/RelWithDebInfo)
  set(ORT_BUILD_COMMAND ./build.sh --config RelWithDebInfo --skip_tests --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(ORT_BUILD_DIR ${ORT_SOURCE_DIR}/build/Windows/RelWithDebInfo)
  set(ORT_BUILD_COMMAND cmd /c build.bat --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync)
endif()

set(ORT_INCLUDE_DIR ${ORT_SOURCE_DIR}/include/onnxruntime/core/session)
set(ORT_LIB ${ORT_BUILD_DIR}/libonnxruntime${LIB_EXT})

ExternalProject_Add(onnxruntime_proj
    PREFIX ${CMAKE_BINARY_DIR}/onnxruntime
    SOURCE_DIR ${ORT_SOURCE_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${ORT_BUILD_COMMAND}
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${ORT_LIB}
)

# Copy ONNX Runtime .so to deploy dir
ExternalProject_Add_Step(onnxruntime_proj copy_lib
    DEPENDEES build
    COMMAND ${CMAKE_COMMAND} -E copy ${ORT_LIB} ${DEPLOY_DIR}/
    COMMENT "Copy ONNX Runtime shared library to deploy folder"
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
    ${ORT_LIB}
    ${OPENCV_CORE_LIB}
    ${OPENCV_IMGPROC_LIB}
    ${OPENCV_IMGCODECS_LIB}
    ${OPENCV_DNN_LIB}
    #${OPENCV_HIGHGUI_LIB}
)

# Set RPATH so the runtime linker finds the .so files in deploy dir
set_target_properties(my_app PROPERTIES
    BUILD_RPATH "${DEPLOY_DIR}"
    INSTALL_RPATH "${DEPLOY_DIR}"
)
