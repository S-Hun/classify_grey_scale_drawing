# '__future': python 2에서 python 3 문법 사용
from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

# 텐서플로우 및 케라스 라이브러리 임포트
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

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

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

batch_size = 128
epochs = 10

model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, Y_test))

################################ 성능 검증, 평가 ################################

# 모델 성능 검증
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print('\n테스트 정확도: ', test_acc)

################################ 모델 및 데이터 이름 저장 ################################

import pickle

model.save('222222222222222222222222222' + '.h5')

with open(classes, 'wb') as fp:
    pickle.dump('1111111111111111', fp)