import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
'''
# import dataset
mnist = tf.keras.datasets.mnist
#split into training and testing using in-built load data function
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalise so that all values fall between 0 and 1 for ease of use
# we do not want to normalise the digits but only the pixels 
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

# add layer, in this case flatten turns 2d dataset into a single 784 dimension array
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

#this adds a dense layer with 128 nodes. in a dense layer, each previous node is connected to each node
# we add this layer in conjunction with an activation function. in this case 'relu'
# relu is rectify linear unit, starts negative at 0 and then increases linearly
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

#output layer
#10 nodes since 0-9 as options
# softmax ensures that sum of all outputs adds to 1
# this is almost a probability of answers 0-9
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#epochs are how many iterations, how many times model sees same data
model.fit(x_train, y_train, epochs=3)

model.save('first.attempt')
'''
