# yolov8-onnx-ise
Deploy YoloV8 on Onnxruntime with custom RISC-V ISE

# Getting started

Update submodules
``` sh
git submodule update --init --recursive
```

## Install Python dependencies
``` sh
pip install -r requirements.txt
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
[Netron](https://netron.app) can be used to visualize the computation graph.

