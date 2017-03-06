import argparse
import numpy as np
import six
import time

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L

from batch_normalization import BatchNormalizationFunction
from convolution_2d import Convolution2DFunction
from linear import LinearFunction

class CNN(chainer.Chain):

    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 16, 5),
            conv2=L.Convolution2D(16, 32, 3, pad=1),
            conv3=L.Convolution2D(32, 64, 3, pad=1),
            bn1=L.BatchNormalization(16),
            bn2=L.BatchNormalization(32),
            bn3=L.BatchNormalization(64),
            fc4=L.Linear(3 * 3 * 64, 128),
            fc5=L.Linear(128, 10),
        )

    def __call__(self, x, train=True):
        h = self.conv1(x)
        h = self.bn1(h, test=not train)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.conv2(h)
        h = self.bn2(h, test=not train)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.conv3(h)
        h = self.bn3(h, test=not train)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.fc4(h)
        h = F.relu(h)
        h = self.fc5(h)
        return h

    def forward(self, x, train=True):
        h, = Convolution2DFunction().forward((x, self.conv1.W.data, self.conv1.b.data))
        bn = self.bn1
        h, = BatchNormalizationFunction(bn.eps, None, None, False, 0.0).forward((h, bn.gamma.data, bn.beta.data, bn.avg_mean, bn.avg_var))
        h, = F.ReLU().forward((h,))
        h, = F.MaxPooling2D(2).forward((h,))
        h, = Convolution2DFunction(pad=1).forward((h, self.conv2.W.data, self.conv2.b.data))
        bn = self.bn2
        h, = BatchNormalizationFunction(bn.eps, None, None, False, 0.0).forward((h, bn.gamma.data, bn.beta.data, bn.avg_mean, bn.avg_var))
        h, = F.ReLU().forward((h,))
        h, = F.MaxPooling2D(2).forward((h,))
        h, = Convolution2DFunction(pad=1).forward((h, self.conv3.W.data, self.conv3.b.data))
        bn = self.bn3
        h, = BatchNormalizationFunction(bn.eps, None, None, False, 0.0).forward((h, bn.gamma.data, bn.beta.data, bn.avg_mean, bn.avg_var))
        h, = F.ReLU().forward((h,))
        h, = F.MaxPooling2D(2).forward((h,))
        h, = LinearFunction().forward((h, self.fc4.W.data, self.fc4.b.data))
        h, = F.ReLU().forward((h,))
        h, = LinearFunction().forward((h, self.fc5.W.data, self.fc5.b.data))
        return h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default='-1', help='GPU device ID, negative value indicates CPU')
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Mini batch size')
    parser.add_argument('--iteration', '-i', type=int, default=100, help='The number of iterations')
    args = parser.parse_args()

    net = CNN()

    batch_size = args.batch_size
    iteration = args.iteration
    device_id = args.gpu
    if device_id >= 0:
        cuda.get_device(device_id).use()
        net.to_gpu(device_id)

    xp = net.xp
    x = xp.random.random((batch_size * iteration, 1, 28, 28)).astype(np.float32)
    y = xp.zeros((batch_size * iteration, 10), dtype=np.float32)
    z = xp.zeros((batch_size * iteration, 10), dtype=np.float32)
    print('Normal chainer.function call')
    start_clock = time.clock()
    with chainer.no_backprop_mode():
        for i in six.moves.xrange(0, batch_size * iteration, batch_size):
            y = net(x[i:i + batch_size], train=False)
    print(time.clock() - start_clock)
    print('Direct chainer.function call')
    start_clock = time.clock()
    for i in six.moves.xrange(0, batch_size * iteration, batch_size):
            z = net.forward(x[i:i + batch_size], train=False)
    print(time.clock() - start_clock)
    print('Diff')
    print(xp.mean((z - y.data) ** 2) ** 0.5)

if __name__ == '__main__':
    main()
