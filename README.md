# yolov8-onnx-ise
Deploy YoloV8 on Onnxruntime with custom RISC-V ISE

# Getting started

Update submodules
``` sh
git submodules update --init --recursive
```

## Install Python dependencies
``` sh
pip install -r requirements.txt
```

## Compile ONNX Runtime for host

### Linux
``` sh
cd 3rdparty/onnx
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
```

### MacOS
``` sh
cd 3rdparty/onnx
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64 --use_vcpkg
```

## Prepare YOLO model
``` sh
python scripts/get_yolo.py
```

## Compile application
``` sh
mkdir build
cd build
cmake ..
make
```

# Visualize model graph
[Netron](https:netron.app) can be used to visualize the computation graph.

