@ECHO OFF
ECHO dataset from 'google-quickdraw' which link 'https://github.com/googlecreativelab/quickdraw-dataset'
ECHO You need to install 'gsutil tools' on OS
ECHO press any key to start install
gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy .
PAUSE