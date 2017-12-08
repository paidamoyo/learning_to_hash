import numpy as np
import scipy.io as sio


def generate(data):
    if data == 'mnsit':
        train_data = sio.loadmat('dataset/mnist_training.mat')
        train_x = train_data['Xtraining']
        test_data = sio.loadmat('dataset/mnist_test.mat')
        test_x = test_data['Xtest']
        data = {'test_x': test_x, 'train_x': train_x}
        print("test_x:{}, train_x:{}".format(test_x.shape, train_x.shape))
        return data
    elif data == 'cifar':
        cifar = 'dataset/cifar-10-batches-mat/'
        train_x = np.empty(shape=(0, 3072))
        for idx in range(1, 6):
            batch = get_batch(cifar, idx)
            print("batch:{}".format(batch.shape))
            train_x = np.concatenate([train_x, batch])
        test_data = sio.loadmat(cifar + 'test_batch.mat')
        test_x = test_data['data']
        print("test_x:{}, train_x:{}".format(test_x.shape, train_x.shape))
        data = {'test_x': test_x, 'train_x': train_x}
        return data


def get_batch(cifar, number):
    train_ = sio.loadmat(cifar + 'data_batch_{}.mat'.format(number))
    train_ = train_['data']
    print('batch:{}, train'.format(number, train_.shape))
    return train_


def reshape_cifar(j, test_x):
    return test_x[j].reshape(3, 32, 32).transpose(1, 2, 0)


def reshape_mnsit(j, test_x):
    return test_x[j].reshape(28, 28)