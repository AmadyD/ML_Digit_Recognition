import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Step 2: Preprocess the data
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values


# Step 3: Build a simple Neural Network model
def model_creation(x_train, y_train, x_test, y_test):
    model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 image into a 1D array
    Dense(128, activation='relu'),  # First hidden layer
    Dense(64, activation='relu'),   # Second hidden layer
    Dense(10, activation='softmax') # Output layer (10 classes for digits 0-9)
    ])

    # Step 4: Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Step 5: Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    return model
