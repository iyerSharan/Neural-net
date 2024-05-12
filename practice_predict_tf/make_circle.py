import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import  train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs=None):
        if logs.get('accuracy') >= 0.995:
            self.model.stop_training = True
            print("Accuracy target achieved")

callbacks=MyCallback()

x, y = make_circles(n_samples = 10000,
                        noise= 0.05,
                        random_state=26)

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.33, random_state=26)

x_train = np.array(x_train.astype('float32'))
x_test = np.array(x_test.astype('float32'))

print(x_train.shape[:])
print(x_test.shape[:])

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1)  # No activation function for regression
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',  # Mean Squared Error for regression
              metrics=['accuracy'])  # Use Mean Squared Error as a metric

# Train the model
history = model.fit(x_train, y_train, epochs=100, callbacks=[callbacks])

#Validation of the model
validation_loss, validation_accuracy = model.evaluate(x_test, y_test)
print("Validation_loss, Validation_accuracy:", (validation_loss, validation_accuracy))

prediction = model.predict(x_test)
prediction_true = y_test

# Comparison
print("Predictions vs True Labels:")
for i in range(10):
    print("Prediction:", prediction[i], "True Label:", prediction_true[i])







