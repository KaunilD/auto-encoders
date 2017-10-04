from data import Data
from model import Model
import cv2
from chainer import Variable, optimizers, datasets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
model = Model()
lr = 0.001

optimizer = optimizers.SGD(lr = lr)
optimizer.setup(model)

epochs = 200
bs = 100

train, test = datasets.get_mnist()
print(len(train))
x_train = list()
for i in range(0, len(train)):
    x_train.append(train[i][0])
y_train = x_train.copy()
for epoch in range(epochs):
    epoch_sum_loss = 0
    for i in range(0, len(x_train), bs):
        batch_x_data = np.asarray(x_train[i:i+bs])
        batch_y_data = np.asarray(y_train[i:i+bs])

        x = Variable(batch_x_data)
        t = Variable(batch_y_data)

        optimizer.update(model, x, t)
        epoch_sum_loss += model.loss * bs
    model.train = False
    pred = model(Variable(np.asarray([[x_train[0]]])), None)
    pred = np.asarray(pred.data[0])
    pred.shape = (28, 28)
    plt.imshow(pred)
    plt.pause(0.01)
    model.train = True
    print(model.loss.data)
    epoch_avg_loss = epoch_sum_loss / len(x_train)
    print('Epoch: {} Loss: {}'.format(epoch, epoch_avg_loss.data))
