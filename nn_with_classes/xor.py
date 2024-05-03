import numpy as np

from network import Network
from fclayer import FCLayer
from activation import tanh, tanh_prime
from activation_layer import ActivationLayer
from losses import mse, mse_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# set-up the network
net = Network()
net.add(FCLayer(2,3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3,1))
net.add(ActivationLayer(tanh, tanh_prime))

# train the network
net.use(mse, mse_prime)
net.fit(x_train=x_train, y_train=y_train, epoch=1000, learning_rate=0.1)

# test the network
out = net.predict(x_train)
print(out)
