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
# X_train = DS['X_train']
# Y_train = DS['Y_train']
X_test = DS['X_test']
Y_test = DS['Y_test']
classes = DS['classes']

############################### 모델 설계 및 학습 ################################

model = tf.keras.models.load_model('model1-35/20200821/model.h5')

# # Alex Net
# model = Sequential([
#     BatchNormalization(),
#     Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
#     MaxPooling2D(pool_size=(2, 2), padding='same'),
#     BatchNormalization(),
#     Dropout(0.8),
#     Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
#     MaxPooling2D(pool_size=(2, 2), padding='same'),
#     BatchNormalization(),
#     Dropout(0.8),
#     Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
#     MaxPooling2D(pool_size=(2, 2), padding='same'),
#     BatchNormalization(),
#     Dropout(0.8),
#     Flatten(),
#     Dense(1024, activation='relu'),
#     Dense(1024, activation='relu'),
#     Dense(len(classes), activation='softmax')
# ])

# model.compile(optimizer='adam',
#             loss = 'categorical_crossentropy',
#             metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
#             )

# epochs = 30
# model.fit(X_train, Y_train, epochs=epochs, verbose=1)

################################ 성능 검증, 평가 ################################

# 모델 성능 검증
test_loss, test_acc, test_topk  = model.evaluate(X_test, Y_test, verbose=2)
print('\n테스트 정확도: ', test_acc)
print('topK 테스트', test_topk)

################################ 모델 및 데이터 이름 저장 ################################

# import pickle

# model.save('model' + '.h5')

# with open('dump', 'wb') as fp:
#     pickle.dump(list(classes), fp)