import argparse
import tensorflow as tf
import os
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
tf.keras.backend.set_learning_phase(0)
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
import numpy as np
from tensorflow.python.tools import optimize_for_inference_lib
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'



def tflite_convert_speed(input_array, output_array, pb_path, tflite_path):
	converter = tf.lite.TFLiteConverter.from_frozen_graph(pb_path, 
			input_arrays=input_array, 
			output_arrays=output_array  
			)

	converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
	converter.target_spec.supported_types = [tf.float32]
	tfmodel = converter.convert() 
	open (tflite_path , "wb").write(tfmodel)




if __name__ == "__main__":
    filepath = "Digit_Recognition.h5"
    model = tf.keras.models.load_model(filepath)
    print(model.inputs)
    print(model.outputs)

    input_array = [inp.op.name for inp in model.inputs]
    output_array = [out.op.name for out in model.outputs]

    print(input_array)
    print(output_array)

    tflite_convert_speed(input_array, output_array, "Digit_Recognition.pb", "Normal_Tflite_Digit_Recognition.tflite")
