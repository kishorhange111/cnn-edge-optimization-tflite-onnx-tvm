# python3 inference.py --model_type tinyv3 --class_path model_data/coco_classes.txt --model model_data/tiny_v3.h5 --vid vid_file.mp4 --img img_dir --in_size 416

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import numpy as np
import cv2
import time
import tensorflow as tf
import onnxruntime



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--model", required=True,help="saved model path")
    args = vars(ap.parse_args())
    

    image_size = '28'
    model_path = args["model"]
    in_size = int(image_size)

    session = onnxruntime.InferenceSession(model_path)

    (X_train, y_train) , (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 255
    X_train = X_train.reshape(-1,28,28,1)
    img = np.expand_dims(X_train[100], axis=0).astype("float32")
    print(img.shape)
    print(img.shape)
    results = session.run(['dense_output/Softmax'], {"conv1_input": img})
    avg_time = 0
    avg_fps = 0
    for i in range(100):
        tic = time.time()
        results = session.run(['dense_output/Softmax'], {"conv1_input": img})
        toc = time.time()
        print("i: ", i, " : ", ((toc - tic)+0.0000001)*1000, 'ms', " fps: ", 1/((toc - tic)+0.0000001))
        avg_time = avg_time + (((toc-tic)+0.0000001)*1000)
        avg_fps = avg_fps + (1/((toc - tic)+0.0000001))

    print("\navg time for 100 iter: ", (avg_time)/100, 'ms', " fps: ", avg_fps/100)




        