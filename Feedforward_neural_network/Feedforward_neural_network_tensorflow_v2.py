# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:33:00 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

FNN in tensorflow_v2
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import argparse
from collections import OrderedDict

tf.keras.backend.set_floatx('float64')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default =128)
    
    args = parser.parse_args()
    
    config = OrderedDict([
            ('epochs', args.epochs),
            ('batch_size', args.batch_size)])
    
    return config
    
config = parse_args()


### data load ###
features = pd.read_pickle('../datasets/four_features.pickle')
features = np.array(features)
features = scale(features)

train = features[:-365,:]
test = features[-365:,:]


### seperate features and target ###
train_x = np.array(train[:,:3])
train_y = np.array(train[:,3:])

test_x = np.array(test[:,:3])
test_y = np.array(test[:,3:])  

n_samples = len(train_x)

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_x, train_y)).shuffle(n_samples).batch(config['batch_size'])
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(config['batch_size'])

class FNN(Model):
    def __init__(self):
        super(FNN, self).__init__()
        self.d1 = Dense(64, activation='sigmoid')
        self.d2 = Dense(32, activation='sigmoid')
        self.d3 = Dense(1)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)
    

model = FNN()

optimizer = tf.keras.optimizers.Adam(lr=0.01)
loss_object = tf.keras.losses.MeanSquaredError()

train_loss = tf.keras.metrics.Mean(name='train_loss')

test_loss = tf.keras.metrics.Mean(name='test_loss')

@tf.function
def train_step(X, Y):
    with tf.GradientTape() as tape:
        predictions = model(X)
        loss = loss_object(Y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
     
@tf.function
def test_step(X, Y):
    predictions = model(X)
    t_loss = loss_object(Y, predictions)

    test_loss(t_loss)
    

### run ###
def runs():
    for epoch in range(config['epochs']):
        for epoch_x, epoch_y in train_ds:
            train_step(epoch_x, epoch_y)
    
        test_step(test_x, test_y)
    
        template = 'epoch: {}, train_loss: {}, test_loss: {}'
        print(template.format(epoch+1,
                                 train_loss.result(),
                                 test_loss.result()))    

runs()    

### results ###
train_predict_y = model(train_x)
test_predict_y = model(test_x)


### mean squared error ###
train_mse = loss_object(train_y, train_predict_y)
test_mse = loss_object(test_y, test_predict_y)
print('train MSE is %.4f' %(train_mse))
print('test MSE is %.4f' %(test_mse))


### font size ###
plt.rcParams.update({'font.size': 15})


### draw outputs ###
plt.figure(figsize=(15,7))
plt.plot(test_y, label = 'true', c = 'r', marker = '_')
plt.plot(test_predict_y, label = 'prediction', c = 'k')
plt.title('Multi-Layer Perceptron Keras')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend(loc = 1)

