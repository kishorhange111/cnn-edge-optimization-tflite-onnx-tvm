# Deep Neural Network Optimization

Final-year B.Tech (CSE) project at DKTE Society's Textile and Engineering Institute, Ichalkaranji (2022-23).

We took a trained Keras digit-recognition CNN and built a pipeline that converts and compiles it into smaller, faster formats for edge devices. Each version is evaluated to check that accuracy stays in an acceptable range.

## Pipeline

```
Keras (.h5) ──► Frozen graph (.pb) ──► Optimized graph (.pb)
      │                                        │
      └──► TFLite (.tflite) ◄──────────────────┘
                 │
                 └──► ONNX (.onnx) ──► TVM compiler (graph + operator-level tuning)
```

1. **Train** a CNN on handwritten digits (`train.py`)
2. **Freeze** the Keras model to a TensorFlow graph and strip training-only ops (`keras_to_frozen.py`, `pb_to_optimized_pb.py`)
3. **Convert to TFLite**, which applies operator fusion for mobile and embedded targets (`keras_to_tflite.py`, `normal_pb_to_tflite.py`, `optimized_pb_to_tflite.py`)
4. **Convert to ONNX** for framework-independent deployment (`h5_to_onnx.py`, `Tflite_To_Onnx.txt`)
5. **Compile and tune with Apache TVM** (graph-level and operator-level optimizations for the target hardware) in `TVM_COMPILER_TUNING (1).ipynb`
6. **Evaluate** each model (`keras_evaluation.py`, `Normal_Pb_Evaluation.py`, `Tflite_evaluation.py`, `Onnx_Evaluation.py`)

Models were inspected with [Netron](https://netron.app) to compare graphs before and after optimization.

## Model sizes

| Model | File | Size |
|---|---|---|
| Keras | `Digit_Recognition.h5` | 1.50 MB |
| Frozen graph | `Digit_Recognition.pb` | 1.44 MB |
| Optimized graph | `Optimized_pb.pb` | 484 KB |
| TFLite (from Keras) | `Keras_Converted_Tflite_Digit_recognition.tflite` | 482 KB |
| TFLite (from optimized graph) | `Optimized_Digit_Recognition.tflite` | 125 KB |
| ONNX (optimized) | `Optimized_Onnx.onnx` | 125 KB |

The optimized TFLite and ONNX models are about **12x smaller** than the original Keras model.

## Tech stack
`Python` `TensorFlow` `Keras` `TensorFlow Lite` `ONNX` `ONNX Runtime` `Apache TVM` `Netron`

## Report and slides
- [Final Report](Final%20Report%2011-06-23.pdf)
- [Final Presentation](Final%20PPT%2011-06-23.pptx)

## Team
Chetan Bavache · Pranav Bhokare · Akshay Chougale · Omkar Dhanale · Kishor Hange

Guide: Prof. P. M. Gavali, Department of Computer Science & Engineering, DKTE, Ichalkaranji

Originally published at [G11-Mega-Project-2023/Deep-Neural-Network-Optimizations](https://github.com/G11-Mega-Project-2023/Deep-Neural-Network-Optimizations).
