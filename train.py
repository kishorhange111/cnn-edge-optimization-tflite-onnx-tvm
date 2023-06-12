import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

print(tf.__version__)
# exit(0)

(X_train, y_train) , (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1) 

def build_model():
    model = Sequential()
    model.add(Conv2D(25, (3, 3), strides=1, name="conv1", padding='same', activation='relu', kernel_initializer="he_normal",
               input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2), strides=2, padding='same', name="maxpool1"))
    model.add(
        Conv2D(64, (3, 3), strides=1, name="conv2", padding='same', activation='relu', kernel_initializer="he_normal"))
    model.add(MaxPooling2D((2, 2), strides=2, name="maxpool2", padding='same'))
    model.add(
        Conv2D(64, (3, 3), strides=1, name="conv3", padding='same', activation='relu', kernel_initializer="he_normal"))
    model.add(MaxPooling2D((2, 2), strides=2, name="maxpool3", padding='same'))
    model.add(Flatten(name="flatten"))
    model.add(Dense(64, name="dense1", activation='relu'))
    model.add(Dense(32, name="dense2", activation='relu'))
    model.add(Dense(10, name="dense_output", activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file="model_enhanced.png", show_shapes=True)
    return model


model = build_model()
model.fit(X_train,y_train,epochs=2)

model.save("Digit_Recognition.h5")

