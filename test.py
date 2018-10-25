import numpy as np
import scipy.ndimage
import gzip
import pickle
import netCDF4

# read data
with gzip.open('OCEAN_data/data.pkl.gz') as fp:
    data = np.array(pickle.load(fp)).astype(float)

# clean data
data[data == -1.0E20] = 0

# read sst_ave
file_s = netCDF4.Dataset('OCEAN_data/sst_ave300Y.nc')

MONTHS = {
    0: "sst_jan_ave",
    1: "sst_feb_ave",
    2: "sst_mar_ave",
    3: "sst_april_ave",
    4: "sst_may_ave",
    5: "sst_june_ave",
    6: "sst_july_ave",
    7: "sst_agu_ave",
    8: "sst_sep_ave",
    9: "sst_oct_ave",
    10: "sst_nov_ave",
    11: "sst_dec_ave"
}

sst_ave = []
for i in range(12):
    sst_ave.append(file_s.variables[MONTHS[i]][:, :])

file_s.close()


# re-build
data = np.array(data)
sst_ave = np.array(sst_ave)

# prepare
index = []
redata = []
for i in range(len(data)):
    tem = data[i].reshape(200, 360)
    re = tem[20:170, 40:200]
    re = re.reshape(1, 960)
    redata.append(re)
    nino = tem - sst_ave[i % 12]
    nino = nino[62:130, 129:189]
    nino = np.mean(nino)
    index.append(nino)

index = np.array(index)
redata = np.array(redata)


print(redata.shape)
print(index.shape)
with gzip.open('OCEAN_data/redata.pkl.gz', 'wb') as f:
    f.write(pickle.dumps(redata))

with gzip.open('OCEAN_data/index.pkl.gz', 'wb') as f:
    f.write(pickle.dumps(index))

# nino = np.sum(predict - np.sum(predict, axis=0).reshape(1, predict.shape[1]) / predict.shape[0], axis=1) \
#            .reshape(predict.shape[0], 1) / predict.shape[1]
# start = preprocessing.scale(start)
# ALL = None
# for i in start:
#     i.reshape(150, 160)
#     i = scipy.ndimage.zoom(i.reshape(150, 160), 0.2)
#     i = i.reshape(1, 960)
#     if ALL is None:
#         ALL = i
#     else:
#         ALL = np.vstack((ALL, i))
# X = ALL[0:200]
# Y = nino[0:200]
# Xt = ALL[201:300]
# Yt = nino[201:300]
# print(X.shape)
# print(Y.shape)


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# X = np.vstack((mnist.train.images.astype(float), mnist.validation.images.astype('float')))
# Y = np.vstack((np.argmax(mnist.train.labels, 1)[:, None],
#                np.argmax(mnist.validation.labels, 1)[:, None]))
# Xt = mnist.test.images.astype(float)
# Yt = np.argmax(mnist.test.labels, 1)[:, None]
#
# print(X[0:10])
# print(Y[0:10])

