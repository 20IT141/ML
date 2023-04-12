import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
digits = load_digits()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Reshape the data to 8x8 images
X_train = X_train.reshape(X_train.shape[0], 8, 8, 1)
X_test = X_test.reshape(X_test.shape[0], 8, 8, 1)

# Convert the labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Define the first CNN model
model1 = keras.models.Sequential([
    keras.layers.Conv2D(2, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(4, kernel_size=(3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

# Define the second CNN model
model2 = keras.models.Sequential([
    keras.layers.Conv2D(4, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(8, 8, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(2, kernel_size=(3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the models with categorical crossentropy loss and Adam optimizer
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the models on the training set
model1.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
model2.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
