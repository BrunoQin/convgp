import numpy as np
from sklearn import preprocessing
import scipy.ndimage
import gzip
import pickle


# read data
with gzip.open('OCEAN_data/predict.pkl.gz') as fp:
    predict = np.array(pickle.load(fp)).astype(float)

with gzip.open('OCEAN_data/start.pkl.gz') as fp:
    start = np.array(pickle.load(fp))

# clean data
start[start == -1.0E20] = 0
# start = start[:, ~np.all(np.isnan(start), axis=0)]
# start_ave = np.mean(start, axis=0)
# start = start - start_ave

predict[predict == -1.0E20] = 0
# predict = predict[:, ~np.all(np.isnan(predict), axis=0)]
# predict_ave = np.mean(predict, axis=0)
# predict = predict - predict_ave

nino = np.sum(predict - np.sum(predict, axis=0).reshape(1, predict.shape[1]) / predict.shape[0], axis=1)\
           .reshape(predict.shape[0], 1) / predict.shape[1]

start = preprocessing.scale(start)
predict = preprocessing.scale(predict)

# print(nino[0:10])
# print(start[0:10])

start[0].reshape(150, 160)
print(scipy.ndimage.zoom(start[0].reshape(150, 160), 0.2))
print(scipy.ndimage.zoom(start[0].reshape(150, 160), 0.2).shape)


X = None
for i in start:
    i.reshape(150, 160)
    i = scipy.ndimage.zoom(i.reshape(150, 160), 0.2)
    i = i.reshape(1, 960)
    if X is None:
        X = i
    else:
        X = np.vstack((X, i))

X = np.array(X)
print(X.shape)


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# X = np.vstack((mnist.train.images.astype(float), mnist.validation.images.astype('float')))
# Y = np.vstack((np.argmax(mnist.train.labels, 1)[:, None],
#                np.argmax(mnist.validation.labels, 1)[:, None]))
# Xt = mnist.test.images.astype(float)
# Yt = np.argmax(mnist.test.labels, 1)[:, None]
#
# print(X[0:10])
# print(Y[0:10])

