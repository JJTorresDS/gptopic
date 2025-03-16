from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

#model
model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.3),
    #keras.layers.Dense(10, activation = 'softmax'),
    
])

print(model.summary())

#loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss = loss, optimizer = optim, metrics = metrics)

#training
batch_size = 64
epochs = 20

model.fit(x_train, y_train, batch_size = batch_size, epochs= epochs, shuffle=True, verbose=2)

#evaluate
print(model.evaluate(x_test, y_test, batch_size = batch_size, verbose=2))