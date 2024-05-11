import tensorflow as tf
import numpy as np
from tensorflow import keras

# To estimate the value of the function 1/(1-e^(-x)))
x_in=np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
y_in=np.array([0.119, 0.269, 0.5, 0.731, 0.881])

# Reshape input data to match the expected input shape of the model
x_reshaped = x_in.reshape(-1, 1)  # Reshape x to a column vector
y_reshaped = y_in.reshape(-1, 1)  # Reshape y to a column vector
input_data = np.concatenate((x_reshaped, y_reshaped), axis=1)  # Concatenate x and y as input features


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
model.fit(input_data, y_in, epochs=5000)
# Value to predict
x_to_predict = np.array([3.0])
y_to_predict = np.array([0.0])  # Placeholder for prediction, since y value will be predicted

# Reshape input data to match the expected input shape of the model
x_to_predict_reshaped = x_to_predict.reshape(-1, 1)  # Reshape x to a column vector
y_to_predict_reshaped = y_to_predict.reshape(-1, 1)  # Reshape y to a column vector
input_to_predict = np.concatenate((x_to_predict_reshaped, y_to_predict_reshaped), axis=1)  # Concatenate x and y as input features

# Predict
y_predicted = model.predict(input_to_predict)

print("Predicted y value for x = 3.0:", y_predicted[0, 0])



# Print model summary
#model.summary()