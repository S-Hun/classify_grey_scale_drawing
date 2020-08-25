from keras import models, layers
from keras import Model
from keras.models import Sequential
from keras import optimizers, losses ,initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, ReLU, Dense, Flatten
from keras.layers import Conv2D, GlobalAveragePooling2D, ZeroPadding2D, MaxPool2D

import tensorflow as tf
import os
import numpy as np
import math

DS = np.load('Dataset.npz')
X_train = DS['X_train']
Y_train = DS['Y_train']
X_test = DS['X_test']
Y_test = DS['Y_test']
classes = DS['classes']

batch_size = 1
epoches = 30
K = len(classes)

class ResidualUnit(Model):
    def __init__(self, filter_in, filter_out, kernel_size):
        super(ResidualUnit, self).__init__()

        self.bn = BatchNormalization()
        self.conv = Conv2D(filter_out, kernel_size, padding='same')

        if filter_in == filter_out:
            self.identify = lambda x:x
        else:
            self.identify = Conv2D(filter_out, (1,1), padding='same')
    
    def call(self, x, training=False, mask=None):
        h = self.bn(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv(h)

        h = self.bn(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv(h)
        return self.identify(x) + h

class ResnetLayer(Model):
    def __init__(self, filter_in, filters, kernel_size):
        super(ResnetLayer, self).__init__()
        self.sequence = list()

        for f_in, f_out in zip([filter_in] + list(filters), filters):
            self.sequence.append(ResidualUnit(f_in, f_out, kernel_size))

    def call(self, x, training=False, mask=None):
        for unit in self.sequence:
            x = unit(x, training=training)
        return x

class ResNet(Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = Conv2D(8, (3,3), padding='same', activation='relu')

        self.res1 = ResnetLayer(8, (16, 16), (3, 3))
        self.pool1 = MaxPool2D((2, 2))
        self.res2 = ResnetLayer(16, (32, 32), (3, 3))
        self.pool2 = MaxPool2D((2, 2))
        self.res3 = ResnetLayer(32, (64, 64), (3, 3))
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(K, activation='softmax')
    
    def call(self, x, training=False, mask=None):
        x = self.conv(x)

        x = self.res1(x, training=training)
        x = self.pool1(x)
        x = self.res2(x, training=training)
        x = self.pool2(x)
        x = self.res3(x, training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images, training=False)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

# mnist = tf.keras.datasets.mnist

# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# X_train, X_test = X_train / 255.0, X_test / 255.0

# X_train = X_train[..., tf.newaxis].astype(np.float32)
# X_test = X_test[..., tf.newaxis].astype(np.float32)

# X와 Y로 데이터프레임을 만드는 법
train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(batch_size)

model = ResNet()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = optimizers.Adam()

train_loss = metrics.Mean(name='train_loss')
train_accuracy = metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = metrics.Mean(name='test_loss')
test_accuracy = metrics.SparseCategoricalAccuracy(name='test_accuracy')

for epoch in range(epoches):
    for images, labels in train_ds:
        train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)
    
    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)
    
    template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))