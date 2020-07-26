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

################################ 여기까지 라이브러리 호출 ################################

def train_with_cnn(path, save_model_name = None, result_list_name = 'dump'):
    dataset_name = []
    train_len = []
    test_len = []
    train_labels = []
    test_labels = []

    file_list = os.listdir(path)
    i = -1

    train_test_scope = 10

    for npy_file in file_list:
        if 'npy' not in npy_file:
            continue
        i += 1
        print(i, ": Get", npy_file, "...")
        images = np.load(path + npy_file, encoding='latin1', allow_pickle=True)
        dataset_name.append(npy_file.replace('.npy', '').replace('full_numpy_bitmap_', ''))
        images_len = len(images)
        train_len.append(images_len - (images_len // train_test_scope))
        test_len.append(images_len // train_test_scope)
        if i == 0: 
            pri_train_images = images[0:train_len[i]]
            pri_test_images = images[train_len[i]:images_len]
        else: 
            pri_train_images = np.concatenate((pri_train_images, images[0:train_len[i]]), axis=0)
            pri_test_images = np.concatenate((pri_test_images, images[0:test_len[i]]), axis=0)
        train_labels += [i] * train_len[i]
        test_labels += [i] * test_len[i]

    # Transformate to one-hot

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # building the input vector from the 28x28 pixels
    train_images = pri_train_images.reshape(pri_train_images.shape[0], 28, 28, 1)
    test_images = pri_test_images.reshape(pri_test_images.shape[0], 28, 28, 1)
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # one-hot encoding using keras' numpy-related utilities
    n_classes = len(dataset_name)
    print("Shape before one-hot encoding: ", train_labels.shape)
    Y_train = keras.utils.to_categorical(train_labels, n_classes)
    Y_test = keras.utils.to_categorical(test_labels, n_classes)
    print("Shape after one-hot encoding: ", Y_train.shape)

    # 스케일링
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print("End processing")

    ############################### 여기까지 전처리 ################################

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
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    batch_size = 128
    epochs = 1

    model.fit(train_images, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(test_images, Y_test))

    ################################ 여기까지 학습 ################################

    # 모델 성능 검증
    test_loss, test_acc = model.evaluate(test_images, Y_test, verbose=2)
    print('\n테스트 정확도: ', test_acc)

    ################################ 여기까지 성능 검증 ################################
    if save_model_name:

        import pickle

        model.save(save_model_name + '.h5')

        with open(result_list_name, 'wb') as fp:
            pickle.dump(dataset_name, fp)

    ############################### 여기까지 모델 및 데이터 이름 저장 ################################

if type(sys.argv) is not type([]) or len(sys.argv) <= 1:
    print("\nINPUT FORMAT ERROR OCCUR!!!\n")
    print("Input like below(without square-bracket)")
    print("python " + sys.argv[0] + " [dataset address]" + " ([model name] [result list name])")
else:
    print("use " + sys.argv[1] + " as dataset")
    if len(sys.argv) > 2:
        print("save model as " + sys.argv[2])
    if len(sys.argv) > 3:
        print("save result name as " + sys.argv[3])
    train_with_cnn('./dataset', 'model', 'dump')