import sys
import numpy as np
from scipy import stats
from data_processing import n_runs, n_cats
from utils import *


def rdm_corr(a, b, flatten=True):
    if flatten:
        a = flatten_matrix(a)
        b = flatten_matrix(b)
    corr = np.corrcoef(a, b)[0, 1]
    return corr


def permutation_test(model, gt):
    n = 10000
    random_dist = np.zeros(n)
    gt = flatten_matrix(gt)
    model = flatten_matrix(model)
    model_random = model.copy()

    for i in range(n):
        np.random.shuffle(model_random)
        random_corr = rdm_corr(model_random, gt, flatten=False)
        random_dist[i] = random_corr

    r = rdm_corr(model, gt, flatten=False)
    p = np.sum(random_dist > r) / n

    return r, p, random_dist


def model_distribution(model_runs, gt):
    n = 10000
    random_dist = np.zeros(n)
    gt = flatten_matrix(gt)

    def sample_model():
        model_sample = []
        for i in range(n_cats):
            for j in range(i + 1, n_cats):
                distances = model_runs[i*n_runs:(i+1)*n_runs, j*n_runs:(j+1)*n_runs]
                sample = np.random.choice(distances.flatten(), 1)[0]
                model_sample.append(sample)
        model_sample = np.array(model_sample, dtype=np.float64)
        return model_sample

    for i in range(n):
        model_sample = sample_model()
        corr_sample = rdm_corr(model_sample, gt, flatten=False)
        random_dist[i] = corr_sample

    return random_dist


if __name__ == '__main__':
    assert len(sys.argv) == 2
    subj = int(sys.argv[1])

    gt_rdm = np.load('data/gt_rdm.npy')

    # Metric mean significance p-tests
    for metric in ['emd', 'euclidean', 'corrinv']:
        mean_rdm = np.load('results/s{:02d}/{}_rdm.npz'.format(subj, metric))['mean']
        r, p, dist = permutation_test(mean_rdm, gt_rdm)
        np.savez('results/s{:02d}/{}_perm.npz'.format(subj, metric), r=r, p=p, dist=dist)

    # Metric distributions
    dists = []
    for metric in ['emd', 'euclidean', 'corrinv']:
        rdm = np.load('results/s{:02d}/{}_rdm.npz'.format(subj, metric))['runs']
        dist = model_distribution(rdm, gt_rdm)
        dists.append(dist)
        np.save('results/s{:02d}/{}_distribution.npy'.format(subj, metric), dist)

    # Metric distribution pairwise comparisons
    _, p_emd_euclidean = stats.ttest_ind(dists[0], dists[1], equal_var=False)
    _, p_emd_corrinv = stats.ttest_ind(dists[0], dists[2], equal_var=False)
    np.savez('results/s{:02d}/comparisons.npz'.format(subj),
             p_emd_euclidean=p_emd_euclidean, p_emd_corrinv=p_emd_corrinv)
