import sys
import numpy as np
import ot
from tqdm import tqdm
from data_processing import *


def pairwise_distances(feats, mask, metric='emd'):
    """
    Creates a matrix of the pairwise distances between a list of brain patterns masked to an ROI
    :param feats: List of brain patterns for different stimuli and/or runs
    :param mask: Binary mask for a given ROI
    :param metric: What distance metric to use in order to compare brain patterns across conditions
    :return: (ndarray) matrix containing the distances between each pair of brain patterns
    """
    print('Computing pairwise distances with metric: {}'.format(metric))

    # Select the appropriate distance function to use between brain patterns
    if metric == 'emd':
        f = emd
    elif metric == 'euclidean':
        f = euclidean
    elif metric == 'corrinv':
        f = corrinv
    else:
        raise NotImplementedError('Unimplemented distance metric: {}'.format(metric))

    # Apply the distance metric between each pair of features
    d = [[f(feats[i], feats[j], mask)
          for i in range(len(feats))] for j in tqdm(range(len(feats)))]
    d = np.array(d)
    return d


def emd(feats_i, feats_j, mask):
    feats_i, feats_j = feats_i.copy(), feats_j.copy()

    # Build the distance metric for the EMD algorithm
    coordinates = np.argwhere(mask)
    m = ot.dist(coordinates, coordinates, metric='euclidean')
    m /= m.max()

    # Select only features within mask
    feats_i = feats_i[mask]
    feats_j = feats_j[mask]

    # Make positive normalized histograms
    min_val = min(feats_i.min(), feats_j.min())
    if min_val < 0:
        feats_i -= min_val
        feats_j -= min_val
    feats_i /= feats_i.sum()
    feats_j /= feats_j.sum()

    d = ot.emd2(feats_i, feats_j, m)
    return d


def euclidean(feats_i, feats_j, mask):
    feats_i, feats_j = feats_i.copy(), feats_j.copy()

    # Select only features within mask
    feats_i = feats_i[mask]
    feats_j = feats_j[mask]

    d = np.linalg.norm(feats_i - feats_j)
    return d


def corrinv(feats_i, feats_j, mask):
    feats_i, feats_j = feats_i.copy(), feats_j.copy()

    # Select only features within mask
    feats_i = feats_i[mask]
    feats_j = feats_j[mask]

    d = 1 - np.corrcoef(feats_i, feats_j)[0, 1]
    return d


if __name__ == '__main__':
    assert len(sys.argv) == 2
    subj = int(sys.argv[1])

    brain, mask = read_subject(subj)

    # Mean within-category activities
    mean_brain = []
    for i in range(n_cats):
        start = i * n_runs
        end = (i + 1) * n_runs
        mean_brain.append(np.mean(brain[start:end], axis=0))

    for metric in ['emd', 'euclidean', 'corrinv']:
        mean_d = pairwise_distances(mean_brain, mask, metric)   # RDM for mean patterns across all runs of the stimuli
        d = pairwise_distances(brain, mask, metric)             # RDM for all stimuli across all runs
        np.savez('results/s{:02d}/{}_rdm.npz'.format(subj, metric), mean=mean_d, runs=d)    # Save RDMs
