import numpy as np
from sklearn import preprocessing
import scipy.ndimage
import gzip
import pickle


with gzip.open('OCEAN_data/predict.pkl.gz') as fp:
    predict = np.array(pickle.load(fp)).astype(float)

with gzip.open('OCEAN_data/start.pkl.gz') as fp:
    start = np.array(pickle.load(fp)).astype(float)

    # clean data
start[start == -1.0E20] = 0
predict[predict == -1.0E20] = 0

nino = np.sum(predict - np.sum(predict, axis=0).reshape(1, predict.shape[1]) / predict.shape[0], axis=1) \
           .reshape(predict.shape[0], 1) / predict.shape[1]
start = preprocessing.scale(start)
ALL = None
for i in start:
    i.reshape(150, 160)
    i = scipy.ndimage.zoom(i.reshape(150, 160), 0.2)
    i = i.reshape(1, 960)
    if ALL is None:
        ALL = i
    else:
        ALL = np.vstack((ALL, i))
X = ALL[0:200]
Y = nino[0:200]
Xt = ALL[201:300]
Yt = nino[201:300]

print(Yt)

with gzip.open('results/result.pkl.gz') as fp:
    pY = np.array(pickle.load(fp)).astype(float)


print(pY)

