from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.vgg16 import VGG16
import onnx
import os
os.environ['TF_KERAS'] = '1'
import keras2onnx

onnx_model_name = 'embedding_developer_model.onnx'
model = VGG16()
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, onnx_model_name)