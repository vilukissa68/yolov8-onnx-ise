# yolov8-onnx-ise
Deploy YoloV8 on Onnxruntime with custom RISC-V ISE

# Getting started

Update submodules
``` sh
git submodules update --init --recursive
```

## Compile ONNX Runtime

### Linux

``` sh
cd 3rdparty/onnx
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
```

### MacOS
``` sh
cd 3rdparty/onnx
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64
```


