import numpy as np
import scipy.io as sio

from utils.generate_data import reshape_cifar, reshape_mnsit
from utils.metrics import plot_compare_recall, plot_recon


def compare_recall():
    baeh_recall = np.squeeze(baeh['test_recall'])
    sgh_recall = np.squeeze(sgh['test_recall'])
    print("bah_recall:{}, sgh_recall:{}".format(baeh_recall.shape, sgh_recall.shape))
    plot_compare_recall(baeh=baeh_recall, sgh=sgh_recall, bits=bits)
    return


def compare_templates():
    size = 30
    test_x = np.squeeze(baeh['test_x'])
    baeh_xhat = np.squeeze(baeh['test_xhat'])
    sgh_xhat = np.squeeze(sgh['test_xhat'])

    if data == 'mnsit':
        template = np.hstack(
            [np.vstack([reshape_mnsit(j, test_x), reshape_mnsit(j, baeh_xhat), reshape_mnsit(j, sgh_xhat)
                        ]) for j in range(size)])
    else:
        template = np.hstack(
            [np.vstack([reshape_cifar(j, test_x),
                        reshape_cifar(j, baeh_xhat),
                        reshape_cifar(j, sgh_xhat)
                        ]) for j in range(size)])

    plot_recon(template=template)
    return


if __name__ == '__main__':
    data = 'mnsit'
    bits = '8'
    baeh = sio.loadmat('results/BAEH_{}_{}bit.mat'.format(data, bits))
    sgh = sio.loadmat('results/SGH_{}_{}bit.mat'.format(data, bits))
    compare_recall()
    compare_templates()
