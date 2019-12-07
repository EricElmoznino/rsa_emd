import sys
import os
import shutil
import numpy as np
import ot
from tqdm import tqdm
from data_processing import *


def pairwise_distances(feats, mask, metric='emd'):
    print('Computing pairwise distances with metric: {}'.format(metric))
    if metric == 'emd':
        f = emd
    elif metric == 'euclidean':
        f = euclidean
    elif metric == 'corrinv':
        f = corrinv
    else:
        raise NotImplementedError('Unimplemented distance metric: {}'.format(metric))

    d = [[f(feats[i], feats[j], mask)
          for i in range(len(feats))] for j in tqdm(range(len(feats)))]
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
        shutil.rmtree('results/s{:02d}/{}'.format(subj, metric), ignore_errors=True)
        os.mkdir('results/s{:02d}/{}'.format(subj, metric))

        mean_d = pairwise_distances(mean_brain, mask, metric)
        np.save('results/s{:02d}/{}/mean_rdm.npy'.format(subj, metric), mean_d)

        d = pairwise_distances(brain, mask, metric)
        np.save('results/s{:02d}/{}/rdm.npy'.format(subj, metric), d)
