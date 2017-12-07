import numpy as np
import scipy.io as sio

from utils.metrics import plot_compare_recall


def compare_recall():
    baeh_recall = np.squeeze(baeh['test_recall'])
    sgh_recall = np.squeeze(sgh['test_recall'])
    print("bah_recall:{}, sgh_recall:{}".format(baeh_recall.shape, sgh_recall.shape))
    plot_compare_recall(baeh=baeh_recall, sgh=sgh_recall, bits=bits)
    return


def compare_templates():
    return


if __name__ == '__main__':
    bits = '32'
    baeh = sio.loadmat('results/BAEH_mnsit_{}bit.mat'.format(bits))
    sgh = sio.loadmat('results/SGH_mnsit_{}bit.mat'.format(bits))
    compare_recall()
