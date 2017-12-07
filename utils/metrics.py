import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fontsize = 18
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

font = {'family': 'normal',
        'weight': 'bold',
        'size': 24}

plt.rc('font', **font)
params = {'legend.fontsize': 'x-large',
          # 'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.

sns.set_style('white')
sns.set_context('paper')
sns.set()
title_fontsize = 18
label_fontsize = 12


def plot_cost(cost):
    plt.plot(np.array(range(len(cost))) * 100, np.array(cost))
    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel('Total Cost', fontsize=fontsize)
    plt.savefig('results/cost')
    return


def recall_n(test_data, train_data, hamming_neighbors=1000):
    # Ground Truth as each querry's K =10  Euclidian nearest neighbours
    # recall at first N hamming neighbours =
    #   fraction of retrieved true nearest neighbours/ total number of true nearest neighbours
    print("test_data:{}, train_data:{}".format(test_data.shape, train_data.shape))
    test_l2_nn = np.load("results/test_l2_nn.npy")
    test_recall = []
    for idx, test_i in enumerate(test_data):
        # Hamming distance
        dist = (train_data - test_i)
        dist = np.sum(dist != 0, axis=1)
        sorted_dist = sorted(range(len(dist)), key=lambda k: dist[k])
        test_i_l2 = test_l2_nn[idx]
        recall_idx = []
        euclidean_nn = len(test_i_l2)
        # Recall caluculation
        for nn in np.arange(hamming_neighbors):
            first_m_sorted_dist = sorted_dist[0:nn]
            count = len(set(first_m_sorted_dist) & set(test_i_l2))
            recall_nn = count / euclidean_nn
            recall_idx.append(recall_nn)
        if idx % 100 == 0:
            print("iteration:{}, recall_idx:{}, min_dist:{}, test_i:{}, train_data:{}, sorted_dist:{}".format(
                idx, len(recall_idx), min(dist), len(test_i), len(train_data[idx]), len(sorted_dist)))
        test_recall.append(recall_idx)
    test_recall = np.array(test_recall)

    mean_recall = np.mean(test_recall, axis=0)
    plot_recall(hamming_neighbors, mean_recall)
    return mean_recall


def plot_recall(hamming_neighbors, mean_recall):
    print("test_recall:{}".format(mean_recall.shape))
    np.save('results/test_recall', mean_recall)
    plt.figure()
    plt.plot(np.arange(hamming_neighbors), mean_recall)
    plt.xlabel('Number of retrieved items', fontsize=fontsize)
    plt.ylabel('Recall', fontsize=fontsize)
    plt.savefig('results/recall')
    return


def plot_compare_recall(baeh, sgh, bits):
    hamming_neighbors = len(baeh)
    plt.figure()
    plt.plot(np.arange(hamming_neighbors), baeh, label='BAEH')
    plt.plot(np.arange(hamming_neighbors), sgh, label='SGH')
    plt.legend(loc='best', fontsize=10)
    plt.xlabel('Number of retrieved items', fontsize=fontsize)
    plt.ylabel('Recall', fontsize=fontsize)
    plt.savefig('results/recall_compare_{}bit'.format(bits))
    return


def euclidean_distance(test_data, train_data, nearest_neighbors=10):
    print("train:{}, test:{}".format(train_data.shape, test_data.shape))
    test_ranked_l2 = []
    for idx, test_i in enumerate(test_data):
        dist = np.linalg.norm(train_data - test_i, axis=1)
        sorted_l2 = sorted(range(len(dist)), key=lambda k: dist[k])
        k_nn = sorted_l2[0:nearest_neighbors]
        test_ranked_l2.append(k_nn)
        if idx % 100 == 0:
            print("iteration:{}, knn:{}, dist:{}".format(idx, len(k_nn), dist))
    np.save("results/test_l2_nn", test_ranked_l2)
    return test_ranked_l2


def plot_recon(template):
    # plt.figure(figsize=(60, 60))
    plt.figure()
    # plt.imshow(template, cmap=mpl.cm.Greys)
    plt.imshow(template)
    plt.axis('off')
    plt.grid('off')
    plt.savefig("results/reconstructed_images")
    return
