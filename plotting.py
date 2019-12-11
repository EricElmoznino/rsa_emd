from matplotlib import pyplot as plt
from scipy.cluster import hierarchy as hc


def plot_rdm(array, labels=None, normalize=True):
    if normalize:
        array /= array.max()
    plt.matshow(array, cmap='jet')
    plt.colorbar()
    if labels is not None:
        plt.xticks(range(len(labels)), labels, rotation=30)
        plt.yticks(range(len(labels)), labels)
    plt.show()


def plot_permutation_test(r, p, random_dist):
    plt.hist(random_dist, bins=40, density=True)
    plt.plot(r, 0.025, 'rD')
    plt.title('r={:.2f}, p={:.2f}'.format(r, p))
    plt.show()


def plot_compare_dists(dist1, dist2, name1, name2):
    # Two overlaid distributions on the same plot
    plt.hist(dist1, bins=40, density=True, alpha=0.5, color='b', label=name1)
    plt.hist(dist2, bins=40, density=True, alpha=0.5, color='r', label=name2)
    plt.legend()
    plt.show()


def plot_dendrogram(model, labels):
    hc.dendrogram(hc.linkage(model), labels=labels)
    plt.show()
