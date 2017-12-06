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
    plt.plot(np.array(range(len(cost))) * 100, np.array(cost) / 500)
    plt.xlabel('Iteration')
    plt.ylabel('Total Cost')
    plt.savefig('results/cost')


def recall_n(test_data, train_data, hamming_neighbors=1000):
    # Ground Truth as each querry's K =10  Euclidian nearest neighbours
    # recall at first N hamming neighbours =
    #   fraction of retrieved true nearest neighbours/ total number of true nearest neighbours
    test_l2_nn = np.load("results/test_l2_nn.npy")
    recall = []
    for idx, test_i in enumerate(test_data):
        # Hamming distance
        test_hamm_dist = []
        dist = (train_data - test_i)
        dist = np.sum(dist, axis=1)
        test_hamm_dist.append(dist)
        sorted_hamm = sorted(range(len(test_hamm_dist)), key=lambda k: test_hamm_dist[k])

        # Recall caluculation
        test_i_l2 = test_l2_nn[idx]
        recall_idx = []
        euclidean_nn = len(test_i_l2)
        for nn in np.arange(hamming_neighbors):
            count = len(set(sorted_hamm[0:nn]) & set(test_i_l2))
            recall_idx.append(count / euclidean_nn)
        recall.append(recall_idx)
    recall = np.array(recall)

    print("recall:{}".format(recall.shape))
    np.save('recall', recall)
    plt.figure()
    plt.plot(np.arange(hamming_neighbors), np.mean(recall, axis=0))
    plt.xlabel('Number of retrieved items', fontsize=fontsize)
    plt.ylabel('Recall', fontsize=fontsize)
    plt.savefig('results/recall')
    return


def euclidean_distance(test_data, train_data, nearest_neighbors=10):
    train_mean = train_data.mean(axis=0).astype('float64')
    train_var = np.clip(train_data.var(axis=0), 1e-7, np.inf).astype('float64')
    print("train:{}, test:{}".format(train_data.shape, test_data.shape))
    test_data = (test_data - train_mean) / train_var
    train_data = (train_data - train_mean) / train_var
    test_ranked_l2 = []
    for idx, test_i in enumerate(test_data):
        dist = np.linalg.norm(train_data - test_i, axis=1)
        sorted_l2 = sorted(range(len(dist)), key=lambda k: dist[k])
        k_nn = sorted_l2[0:nearest_neighbors]
        test_ranked_l2.append(k_nn)
        if idx % 100 == 0:
            print("iteration:{}, knn:{}".format(idx, len(k_nn)))
    np.save("results/test_l2_nn", test_ranked_l2)
    return test_ranked_l2
