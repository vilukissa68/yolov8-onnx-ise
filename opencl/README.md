# Get Dependencies

## Install Python dependencies
Setting up a virtual environment might be useful.
``` sh
pip install -r requirements.txt
```

## Get source
Get TVM source and submodule and downgrade to version 18.0 (this is neede to use relay's C-backend).
``` sh
git clone https://github.com/apache/tvm tvm-18
cd tvm-18
git checkout origin/v0.18.0
git submodule update --init --recursive
```

## Configure build 
TVM needs to be configured with LLVM-18 and OpenCL support needs to be enabled
``` sh
mkdir build
cd build
cp ../cmake/config.cmake .
echo "set(USE_OPENCL ON)" >> config.cmake
echo "set(USE_LLVM <path-to-llvm18-config> ON)" >> config.cmake
```


## Build TVM

``` sh
cmake .. && cmake --build . --parallel $(nproc)
```

## Install python package
After succesfully compiling TVM with specified options we need to install tvm python package
``` sh
cd <tvm-18-path>/python
pip install -e .
```

More detailed instructions: https://tvm.apache.org/docs/install/from_source.html

# CMake Build
Using CMake is the preferred way to build the process
``` sh
mkdir build
cd build
cmake .. -DUSE_QUANTIZATION=ON
make -j$(nproc)
```

# CMake Build (RISC-V)
For RISC-V build we need to specify cross compilation parameters.
``` sh
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../../riscv_toolchain.cmake
make -j$(nproc)
```


# Manual Build

## Compile OpenCV
Clone OpenCV and compile for platform as library.
In project root:
``` sh
git submodule update --init --recursive
cd 3rdparty/opencv
mkdir build
cd build
cmake -DOPENCV_GENERATE_PKGCONFIG=ON ..
make -j$(nproc)
sudo make install
```

If CMake build fails the project can be built with g++

## Generate C Model
``` sh
python generate_c_model.py
```

## Compile
``` sh
export TVM_HOME=<path-to-tvm-root-dir>
export LD_LIBRARY_PATH=$TVM_HOME/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=yolov8-onnx-ise/3rdparty/opencv/build/lib:$LD_LIBRARY_PATH
```

### Linux
``` sh
g++ run_yolov8_coco.cpp \
    -std=c++17 \
    -I$TVM_HOME/include \
    -I$TVM_HOME/3rdparty/dlpack/include \
    -I$TVM_HOME/3rdparty/dmlc-core/include \
	$(pkg-config --cflags --libs opencv4) \
    -L$TVM_HOME/build \
    -ltvm_runtime \
    -lOpenCL \
    -o yolov8_app
```

### Mac
On linux we need to use clang++ to correctly invoke OpenCL and OpenCV dependencies
``` sh
clang++ run_yolov8.cpp \
    -std=c++17 \
    -I$TVM_HOME/include \
    -I$TVM_HOME/3rdparty/dlpack/include \
    -I$TVM_HOME/3rdparty/dmlc-core/include \
    -L$TVM_HOME/build \
    -ltvm_runtime \
    -framework OpenCL \
	$(pkg-config --cflags --libs opencv4) \
    -o yolov8_app
```

### Run
OpenCL reads DYLD_LIBRARY_PATH for the compiled model library
``` sh
export DYLD_LIBRARY_PATH=<this-folder>:$DYLD_LIBRARY_PATH
```

Run the executable
``` sh
./yolov8_app
```

