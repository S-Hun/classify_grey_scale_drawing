import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageOps

import pickle
import sys
import os
import os.path

# 입력 이미지는 정사각형
# 정규 규격은 28*28이나 자동으로 변환하므로 더 커도 상관 없음
# 굵은 펜으로 해야함

# Shell에서 "python runTest.py [image_path] ./model/trained_model ./model/dump"
# 또는 function 호출로 "main_module('image path', './model/trained_model.h5', './model/dump')"

def main_module(image_path, path1, path2):
    model = keras.models.load_model(path1)

    image = Image.open(image_path).convert('L')
    result_image = image.resize((28,28))
    result_image = ImageOps.invert(result_image)
    image_result = np.array(result_image)

    image_result = image_result / 255

    result = image_result
    result = result.reshape(result.shape[1], 28, 1)
    result = np.array([result])

    plt.figure()
    plt.imshow(image_result)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    with open (path2) as fp:
        dataset_name = pickle.load(fp)

    predictions = model.predict(result)
    result_dict = dict(zip(dataset_name, predictions[0]))
    result_dict = sorted(result_dict.items(), key=(lambda x:x[1]), reverse=True)
    for li in result_dict[0:10]:
        print(li[0] + '.jpg')

if len(sys.argv) != 1:
    if len(sys.argv) < 4:
        print('\n\n유효하지 않은 파라미터\n')
        print('python ' + sys.argv[0] + ' [image_path] [trained_model(.h5)] [data_list_dump_file]')
        sys.exit()
    else:
        if sys.argv[2].find('.') == -1:
            sys.argv[2] = sys.argv[2] + '.h5'

    if not os.path.isfile(sys.argv[1]):
        print('\n"' + sys.argv[1] + '"' + '은 존재하지 않거나 올바르지 않은 경로입니다.')
    elif not os.path.isfile(sys.argv[2]):
        print('\n"' + sys.argv[2] + '"' + '은 존재하지 않거나 올바르지 않은 경로입니다.')
    elif sys.argv[2].find('.h5') == -1:
        print('\n"' + sys.argv[2] + '"' + '은 h5 파일이 아닙니다.')
    elif not os.path.isfile(sys.argv[3]):
        print('\n"' + sys.argv[3] + '"' + '은 존재하지 않거나 올바르지 않은 경로입니다.')
    else:
        main_module(sys.argv[1], sys.argv[2], sys.argv[3])