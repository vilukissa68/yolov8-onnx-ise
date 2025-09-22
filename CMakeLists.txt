cmake_minimum_required(VERSION 3.14)
project(MyONNXApp)

# Option to toggle between builds
option(STATIC_BUILD "Build with static OpenCV/ONNX" OFF)

if(STATIC_BUILD)
    include(${CMAKE_SOURCE_DIR}/CMakeLists_static.cmake)
else()
    include(${CMAKE_SOURCE_DIR}/CMakeLists_shared.cmake)
endif()
