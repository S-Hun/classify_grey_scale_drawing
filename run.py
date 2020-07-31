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

def get_result(image_path: str, path1: str, path2: str) -> list:
    if not os.path.isfile(image_path):
        print('\n"' + image_path + '"' + '은 존재하지 않거나 올바르지 않은 경로입니다.')
    elif not os.path.isfile(path1):
        print('\n"' + path1 + '"' + '은 존재하지 않거나 올바르지 않은 경로입니다.')
    elif path1.find('.h5') == -1:
        print('\n"' + path1 + '"' + '은 h5 파일이 아닙니다.')
    elif not os.path.isfile(path2):
        print('\n"' + path2 + '"' + '은 존재하지 않거나 올바르지 않은 경로입니다.')
    
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

    with open(path2, 'rb') as fp:
        dataset_name = pickle.load(fp)

    predictions = model.predict(result)
    result_dict = dict(zip(dataset_name, predictions[0]))
    result_dict = sorted(result_dict.items(), key=(lambda x:x[1]), reverse=True)
    li = []
    for ol in result_dict[0:10]:
        li.append(ol[0] + '.jpg')
    return li
    

if len(sys.argv) != 1:
    if len(sys.argv) < 4:
        print('\n\n유효하지 않은 파라미터\n')
        print('python ' + sys.argv[0] + ' [image_path] [trained_model(.h5)] [data_list_dump_file]')
        sys.exit()
    else:
        if sys.argv[2].find('.') == -1:
            sys.argv[2] = sys.argv[2] + '.h5'
            main_module(sys.argv[1], sys.argv[2], sys.argv[3])