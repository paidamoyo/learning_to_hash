import numpy as np
import scipy.io as sio

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
    template = np.hstack(
        [np.vstack([test_x[j].reshape(28, 28), baeh_xhat[j].reshape(28, 28), sgh_xhat[j].reshape(28, 28)
                    ]) for j in range(size)])

    plot_recon(template=template)
    return


if __name__ == '__main__':
    bits = '8'
    baeh = sio.loadmat('results/BAEH_mnsit_{}bit.mat'.format(bits))
    sgh = sio.loadmat('results/SGH_mnsit_{}bit.mat'.format(bits))
    compare_recall()
    compare_templates()
