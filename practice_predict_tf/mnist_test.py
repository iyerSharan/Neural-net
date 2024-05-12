import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils

class MyCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, log=None):
        if log.get('accuracy') >= 0.995:
            print("\n Stopping since accuracy achieved...")
            self.model.stop_training = True

callbacks = MyCallbacks()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train.shape:", x_train.shape[0])
print("y_train.shape:", y_train.shape[0])
print("x_test.shape:", x_test.shape[0])
print("y_test.shape:", x_test.shape[0])

x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255

x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')

])
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(x_train, y_train, batch_size=20, epochs=15, callbacks=[callbacks])

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test_loss, Test_accuracy:", (test_loss, test_accuracy))


