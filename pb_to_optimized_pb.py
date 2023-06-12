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
import keras
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def freeze_session_opt(session, keep_var_names=None, outputs=None, inputs=None, clear_devices=True):
	from tensorflow.python.framework.graph_util import convert_variables_to_constants
	from tensorflow.tools.graph_transforms import TransformGraph

	with tf.gfile.GFile("Optimized_pb.pb", "wb") as f:
		inputs = ['conv1_input'] # replace with your input names
		outputs = ['dense_output/Softmax'] # replace with your output names
		graph_def = session.graph.as_graph_def(add_shapes=True)
		graph_def = convert_variables_to_constants(session, graph_def, outputs)
		graph_def = TransformGraph(
			graph_def,
			inputs,
			outputs,
			[
				"remove_nodes(op=Identity, op=CheckNumerics, op=StopGradient)",
				"sort_by_execution_order", # sort by execution order after each transform to ensure correct node ordering
				"remove_attribute(attribute_name=_XlaSeparateCompiledGradients)",
				"remove_attribute(attribute_name=_XlaCompile)",
				"remove_attribute(attribute_name=_XlaScope)",
				"sort_by_execution_order",
				"remove_device",
				"sort_by_execution_order",
				"fold_batch_norms",
				"sort_by_execution_order",
				"fold_old_batch_norms",
				"sort_by_execution_order"
			]
		)
		f.write(graph_def.SerializeToString())



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
	freeze_session_opt(K.get_session(),output_array, input_array)