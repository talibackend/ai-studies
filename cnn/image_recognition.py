import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from tensorflow.keras.datasets import cifar10
import seaborn as sns
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix, accuracy_score

(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# print(train_x.shape)
# print(test_x.shape)

# print(train_x.max())
# print(test_x.max())

train_x = train_x / 255
test_x = test_x / 255

print(train_y.shape)
print(test_y.shape)

model = tf.keras.models.Sequential()

# First convolutional layer
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=[32, 32, 3], padding="same"))

# Second cnn layer
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same", ))

# Pooling layer
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))

# Add third cnn layer
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", ))

# Add Fourth cnn layer
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", ))

# Add Second pooling layer
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))

# Add dropout layer
model.add(tf.keras.layers.Dropout(0.35))

# Flatten layer
model.add(tf.keras.layers.Flatten())

# Dense layer for ann
model.add(tf.keras.layers.Dense(activation="relu", units=128))

# Output layer
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])

model.fit(train_x, train_y, batch_size=10, epochs=5)

print(model.summary())
