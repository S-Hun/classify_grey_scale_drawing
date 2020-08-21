from tensorflow import keras
import numpy as np
import os

file_path = './dataset/'
file_list = os.listdir(file_path)
length = 0

Y_train = []
Y_test = []
Lable = []

train_size = 9000 
test_size = 1000

for name in file_list:
    if 'npy' not in name:
        print(name, 'is not numpy array')
        continue
    ds = np.load(file_path + name, encoding='latin1', allow_pickle=True)
    if len(ds) < train_size + test_size:
        print(name, 'is too short')
        continue
    length = length + 1
    Lable.append(name.replace('.npy', '').replace('full_numpy_bitmap_', ''))
    print('#', length ,'### file name # {:30} ### file size # {:12} ###'.format(name, str(ds.shape)))
    sample_list = np.random.choice(len(ds), size = train_size + test_size, replace=False)
    sample = ds[sample_list]
    sample = np.reshape(sample, (sample.shape[0], 28, 28, 1))
    if length == 1:
        X_train = sample[0:train_size]
        X_test = sample[train_size:train_size + test_size]
    else:
        X_train = np.concatenate((X_train, sample[0:train_size]), axis = 0)
        X_test = np.concatenate((X_test, sample[train_size:train_size + test_size]), axis = 0)
    Y_train += [length - 1] * train_size
    Y_test += [length - 1] * test_size

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = keras.utils.to_categorical(Y_train, length)
Y_test = keras.utils.to_categorical(Y_test, length)

# X_train = X_train / 255.0
# X_test = X_test / 255.0

print('Shape', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, sep='\n')

np.savez('Dataset_test', X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test, classes = Lable)