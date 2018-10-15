import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X = np.vstack((mnist.train.images.astype(float), mnist.validation.images.astype('float')))
Y = np.vstack((np.argmax(mnist.train.labels, 1)[:, None],
               np.argmax(mnist.validation.labels, 1)[:, None]))
Xt = mnist.test.images.astype(float)
Yt = np.argmax(mnist.test.labels, 1)[:, None]
## test
print(X.shape)
