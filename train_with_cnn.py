# '__future': python 2에서 python 3 문법 사용
from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

# 텐서플로우 및 케라스 라이브러리 임포트
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv1D, LSTM, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# 넘파이 및 matplotlib 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

################################ 데이터 호출 ################################

DS = np.load('Dataset.npz')
X_train = DS['X_train']
Y_train = DS['Y_train']
X_test = DS['X_test']
Y_test = DS['Y_test']
classes = DS['classes']

############################### 모델 설계 및 학습 ################################

# Tensorflow 기본 모델
# model = Sequential([
#     Flatten(input_shape=(28, 28, 1)),
#     Dense(128, activation='relu'),
#     Dense(len(classes), activation='softmax')
# ])

# 어느 사이트의 의류 분류 Net
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                     activation='relu',
#                     input_shape=(28, 28, 1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(classes), activation='softmax'))

# Alex Net
model = Sequential([
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    BatchNormalization(),
    Dropout(0.8),
    Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    BatchNormalization(),
    Dropout(0.8),
    Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    BatchNormalization(),
    Dropout(0.8),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(len(classes), activation='softmax')
])

# LSTM to Parse Strokes
# model = Sequential([
#     BatchNormalization(input_shape=(None,) + X_train.shape[2:]),
#     Conv1D(48, (5,)),
#     Dropout(0.3),
#     Conv1D(64, (5,)),
#     Dropout(0.3),
#     Conv1D(96, (5,)),
#     Dropout(0.3),
#     LSTM(128, return_sequences=True),
#     Dropout(0.3),
#     LSTM(128, return_sequences=False),
#     Dropout(0.3),
#     Dense(256),
#     Dense(len(classes), activation='softmax')
# ])

# LSTM processing
# weight_path = "{}_weights.best.hdf5".format('stroke_lstm_model')

# checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
#                             save_best_only=True, mode='min', save_weights_only=True)

# reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', 
#                                     min_delta=0.0001, cooldown=5, min_lr=0.0001)

# early = EarlyStopping(monitor='val_loss',
#                     mode='min',
#                     patience=5)

# callbacks_list = [checkpoint, early, reduceLROnPlat]

# MNIST CNN Net
# model = keras.Sequential([
#     Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', input_shape=(28,28, 1), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
#     Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
#     Flatten(),
#     Dropout(0.9),
#     Dense(1024, activation='relu'),
#     Dense(len(classes), activation='softmax')
# ])

model.compile(optimizer='adam',
            loss = 'categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
            )

# SGD : epoch 10 : 84%
# Nadam : epoch 10 : los/25 acu/85%
# Adam : epoch 10 : los/54 acu/85%

# LSTM model fit
# model.fit(X_train, Y_train,
#             validation_data = (X_test, Y_test),
#             batch_size = 1024,
#             epochs = 30,
#             shuffle = 'batch',
#             callbacks = callbacks_list)

epochs = 30
model.fit(X_train, Y_train, epochs=epochs, verbose=1)

################################ 성능 검증, 평가 ################################

# 모델 성능 검증
test_loss, test_acc, test_topk  = model.evaluate(X_test, Y_test, verbose=2)
print('\n테스트 정확도: ', test_acc)
print('topK 테스트', test_topk)

################################ 모델 및 데이터 이름 저장 ################################

import pickle

model.save('model' + '.h5')

with open('dump', 'wb') as fp:
    pickle.dump(list(classes), fp)