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
import keras


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
	from tensorflow.python.framework.graph_util import convert_variables_to_constants
	graph = session.graph
	with graph.as_default():
		freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
		output_names = output_names or []
		output_names += [v.op.name for v in tf.global_variables()]

		for v in tf.compat.v1.global_variables():
			print(v.op.name)

		# Graph -> GraphDef ProtoBuf
		input_graph_def = graph.as_graph_def()
		if clear_devices:
			for node in input_graph_def.node:
				node.device = ""
		frozen_graph = convert_variables_to_constants(session, input_graph_def,
														output_names, freeze_var_names)
		# frozen_graph = optimize_for_inference_lib.optimize_for_inference(frozen_graph, ['input_1'], ['conv2d_17/BiasAdd','conv2d_20/BiasAdd'], tf.float32.as_datatype_enum)
		return frozen_graph





if __name__ == "__main__":

	filepath = "Digit_Recognition.h5"
	model = tf.keras.models.load_model(filepath)

	print(model.inputs)
	print(model.outputs)
	

	input_array = [inp.op.name for inp in model.inputs]
	output_array = [out.op.name for out in model.outputs]

	print(input_array)
	print(output_array)
	# exit(0)

	# create frozen graph
	frozen_graph = freeze_session(K.get_session(),
                              output_names=output_array)
	tf.train.write_graph(frozen_graph, "./", "Digit_Recognition.pb", as_text=False)

# conv2d_17/BiasAdd','conv2d_20/BiasAdd
# python3 -m tf2onnx.convert --graphdef fixed1.pb --output new_fixed_11.onnx --inputs input_1:0 --outputs conv2d_17/BiasAdd:0,conv2d_20/BiasAdd:0