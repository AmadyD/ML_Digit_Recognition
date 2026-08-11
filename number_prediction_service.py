import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from models import NeuralNetwork

class NumberPredictionService:

    @staticmethod
    def predict(image):

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        # Step 2: Preprocess the data
        x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values

        model = NeuralNetwork.model_creation(x_train, y_train, x_test, y_test)
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        img = cv2.resize(img, (28, 28))  # Resize to match MNIST format
        img = 255 - img  # Invert colors (white background, black digit)

        # Normalize and reshape for prediction
        img = img / 255.0
        img = img.reshape(1, 28, 28)
        # Predict the digit
        prediction = model.predict(img)
        f_x = tf.nn.softmax(prediction)
        print(f_x)
        predicted_digit = np.argmax(prediction)
        print(predicted_digit)
        # Display the drawn digit
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.title(f'Predicted Digit: {f_x.numpy()[0][predicted_digit]* 100:.2f}%) (Digit: {predicted_digit})')
        plt.axis('off')
        plt.show()