import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import numpy as np
import cv2
import time
import tensorflow as tf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--model", required=True,help="saved model path")
    args = vars(ap.parse_args())

    image_size = '28'
    model_path = args["model"]
    in_size = int(image_size)

    with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session() as sess:
            with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()

                tf.import_graph_def(graph_def, input_map=None, return_elements=None,
                    name="", op_dict=None, producer_op_list=None)
                
                l_input = graph.get_tensor_by_name('conv1_input:0')
                l_output1 = graph.get_tensor_by_name('dense_output/BiasAdd:0')
                tf.compat.v1.global_variables_initializer()
                tf_model = None

                
                (X_train, y_train) , (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                X_train = X_train / 255
                X_train = X_train.reshape(-1,28,28,1)
                img = np.expand_dims(X_train[100], axis=0)
                print(img.shape)
                print(img.shape)
                layerwise_hm = sess.run([l_output1], feed_dict={l_input: img})
                print(np.argmax(layerwise_hm))
                print(y_train[100])
                # print(len(layerwise_hm))
                # exit(0)

                avg_time = 0
                avg_fps = 0
                for i in range(100):
                    tic = time.time()
                    layerwise_hm = sess.run([l_output1], feed_dict={l_input: img})
                    toc = time.time()
                    print("i: ", i, " : ", ((toc - tic)+0.0000001)*1000, 'ms', " fps: ", 1/((toc - tic)+0.0000001))
                    avg_time = avg_time + (((toc-tic)+0.0000001)*1000)
                    avg_fps = avg_fps + (1/((toc - tic)+0.0000001))

                print("\navg time for 100 iter: ", (avg_time)/100, 'ms', " fps: ", avg_fps/100)
                print()

                        