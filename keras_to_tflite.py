import tensorflow as tf



converter = tf.lite.TFLiteConverter.from_keras_model_file('Digit_Recognition.h5') 
tfmodel = converter.convert() 
open ('Keras_Converted_Tflite_Digit_recognition.tflite' , "wb") .write(tfmodel)