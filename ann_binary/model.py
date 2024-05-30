import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from tensorflow.keras.datasets import fashion_mnist
import seaborn as sns
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

train_x = train_x / 255
test_x = test_x / 255

train_x = train_x.reshape(-1, 28*28)
test_x = test_x.reshape(-1, 28*28)

train_y = np.copy(train_y)
train_y[train_y != 1] = 0

test_y = np.copy(test_y)
test_y[test_y != 1] = 0

# print(train_x.shape)
# print(train_y.shape)
# print(np.unique(train_y))
# print(np.unique(test_y))
# print(test_x.shape)
# print(test_y.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(activation="relu", units=128, input_shape=(784, )),
    tf.keras.layers.Dense(units=200, activation="relu"),
    tf.keras.layers.Dense(units=150, activation="relu"),
    tf.keras.layers.Dense(units=70, activation="relu"),
    tf.keras.layers.Dropout(0.35), 
    tf.keras.layers.Dense(activation='sigmoid', units=1)
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(x=train_x, y=train_y, epochs=5)

print(model.summary())

predict = model.predict(test_x)
predict = (predict > 0.5).astype("int32").flatten()

print(predict)

cm = confusion_matrix(test_y, predict)

print(cm)
print(accuracy_score(test_y, predict))
print(recall_score(test_y, predict))
print(precision_score(test_y, predict))

