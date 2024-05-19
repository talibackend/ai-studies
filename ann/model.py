import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from tensorflow.keras.datasets import fashion_mnist
import seaborn as sns
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix, accuracy_score

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)

# print(np.max(train_x))
# print(np.min(train_x))
# print(np.average(train_x))
# print(np.mean(train_x))

# print(np.unique(train_y))

class_map = {
    "0" : "Tshirt/Top",
    "1" : "Trouser",
    "2" : "Pull over",
    "3" : "Dress",
    "4" : "Coat",
    "5" : "Sandal",
    "6" : "Shirt",
    "7" : "Sneaker",
    "8" : "Bag",
    "9" : "Ankle Boot",
}

train_x = train_x / 255.0
test_x = test_x / 255.0

# plot.figure(figsize=(8, 6))
# ax = sns.heatmap(train_x[1621])
# plot.show()

train_x = train_x.reshape(-1, 28*28)
test_x = test_x.reshape(-1, 28*28)

print(train_x.shape)
print(test_x.shape)

model = tf.keras.models.Sequential()
#Input & First Hidden layer
model.add(tf.keras.layers.Dense(units=128, activation="relu", input_shape=(784, )))
#Dropout layer, to avoid overfitting
model.add(tf.keras.layers.Dropout(0.35))
#Output layer
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])

model.fit(train_x, train_y, epochs=5)

print(model.summary())

test_loss, test_accuracy = model.evaluate(test_x, test_y)

print(test_loss)
print(test_accuracy)

predicts = model.predict(test_x)
predicts=np.argmax(predicts,axis=1)

print(predicts)

#Confusion Matrix(I remember this from basic ml)

cm = confusion_matrix(test_y, predicts)
print(cm)

acc = accuracy_score(test_y, predicts)
print(acc)