import numpy as np

dt_set = np.load('Dataset.npz')

print(dt_set['X_train'].shape)