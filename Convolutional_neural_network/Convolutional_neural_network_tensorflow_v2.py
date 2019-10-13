# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:00:20 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

CNN in tensorflow_v2
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from collections import OrderedDict

tf.keras.backend.set_floatx('float64')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 10)
    parser.add_argument('--batch_size', type = int, default =128)
    
    args = parser.parse_args()
    
    config = OrderedDict([
            ('epochs', args.epochs),
            ('batch_size', args.batch_size)])
    
    return config
    
config = parse_args()

@tf.function()
def random_jitter(input_image, input_label):
    input_image = tf.image.resize(input_image, [34, 34],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    input_image = tf.image.random_crop(input_image, size=[28, 28, 1])
    
#    if tf.random.uniform(()) > 0.5:    
#        input_image = tf.image.flip_left_right(input_image)
    return input_image, input_label


### data load ###
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x, test_x = train_x / 255.0, test_x / 255.0

train_x = train_x[..., tf.newaxis]
test_x = test_x[..., tf.newaxis]

n_samples = len(train_x)

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_ds = train_ds.map(random_jitter,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.shuffle(n_samples).batch(config['batch_size'])

test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(config['batch_size'])

class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.c1 = Conv2D(32, 3, 1, padding = 'same', activation='relu')
        self.p1 = MaxPool2D(2)
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = CNN()

optimizer = tf.keras.optimizers.Adam(lr=0.01)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(X, Y):
    with tf.GradientTape() as tape:
        predictions = model(X)
        loss = loss_object(Y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(Y, predictions)
     
@tf.function
def test_step(X, Y):
    predictions = model(X)
    t_loss = loss_object(Y, predictions)

    test_loss(t_loss)
    test_accuracy(Y, predictions)

### run ###
def runs():
    for epoch in range(config['epochs']):
        for epoch_x, epoch_y in train_ds:
            train_step(epoch_x, epoch_y)

        for epoch_x, epoch_y in test_ds:
            test_step(epoch_x, epoch_y)    
    
        template = 'epoch: {}, train_loss: {}, train_acc: {}, test_loss: {}, test_acc: {}'
        print(template.format(epoch+1,
                                 train_loss.result(),
                                 train_accuracy.result()*100,
                                 test_loss.result(),
                                 test_accuracy.result()*100))    

runs()    


### results ###
test_predict_y = model(test_x)


### mean squared error ###
test_acc = test_accuracy(test_y, test_predict_y)
print('test acc is %.4f' %(test_acc))


### font size ###
plt.rcParams.update({'font.size': 15})


### draw outputs ###
plt.figure(figsize=(5,5))
plt.imshow(test_x[0].reshape(28,28))
print(np.argmax(test_predict_y[0]))

