# Install TVM

## Install Python dependencies
Setting up a virtual environment might be useful.
``` sh
pip install ultralytics onnx "numpy<2.0"
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
This configures TVM to use MicroTVM runtime, which is needed for baremetal inference. Recent LLVM versions are not supported in TVM 18.0 the support might need to be turned off.
``` sh
mkdir build
cd build
cp ../cmake/config.cmake .
echo "set(USE_MICRO ON)" >> config.cmake
echo "set(USE_MICRO_STANDALONE_RUNTIME ON)" >> config.cmake
echo "set(USE_LLVM OFF)" >> config.cmake
```


## Build TVM

``` sh
cmake .. && cmake --build . --parallel $(nproc)
```

## Install python package
We need to install tvm python package
``` sh
cd <tvm-18-path>/python
pip install -e .
```

More detailed instructions: https://tvm.apache.org/docs/install/from_source.html

# Generate C Model

``` sh
python generate_c_model.py
```

# Compile
``` sh
gcc -o yolov8_app -std=c99 -I. -I./tvm_c_out/runtime/include -I./tvm_c_out/codegen/host/include main.c platform.c tvm_c_out/codegen/host/src/*.c -lm
```


# Run
This runs the compiled model with a single round of inference with dummy data.
``` sh
./yolov8_app
```

