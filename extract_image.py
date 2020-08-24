from PIL import Image, ImageOps
import numpy as np
import os


ds_path = 'D:\quickdraw-dataset'
img_path = 'D:/temp/'
num_of_image = 1

if ds_path[-1] != '/' and ds_path[-1] != '\\':
    ds_path = ds_path + '/'
data_list = os.listdir(ds_path)

i = 0

for pick in data_list:
    name = pick.replace('.npy', '').replace('full_numpy_bitmap_', '')
    print(i, name)
    i += 1
    npList = np.load(ds_path + pick)
    rnList = np.random.choice(len(npList), size = num_of_image, replace = False)
    npList = npList[rnList]
    npImages = np.reshape(npList, (npList.shape[0], 28, 28))
    for npImage in npImages:    
        img = Image.fromarray(npImage, 'L')
        img = ImageOps.invert(img)
        img.save(img_path + name + '.png')
