import sys
import numpy as np
from scipy import stats
from data_processing import n_runs, n_cats
from utils import *


def rdm_corr(a, b, flatten=True):
    # Pearson correlation between two RDMs (a and b)
    if flatten:
        a = flatten_matrix(a)
        b = flatten_matrix(b)
    corr = np.corrcoef(a, b)[0, 1]
    return corr


def permutation_test(model, gt):
    """
    Estimate the null distribution of correlations between RDMs and the "true" model using a permutation test
    where the labels for conditions are randomly shuffled.
    :param model: RDM obtained using a model of dissimilarity between brain patterns
    :param gt: "True" RDM
    :return: tuple(true model correlation, p-value under null distribution, null distribution)
    """
    n = 10000
    random_dist = np.zeros(n)
    gt = flatten_matrix(gt)
    model = flatten_matrix(model)
    model_random = model.copy()

    # For the desired number of samples to estimate the distribution
    for i in range(n):
        np.random.shuffle(model_random)     # Shuffling RDM after flattening is equivalent to shuffling condition labels
        random_corr = rdm_corr(model_random, gt, flatten=False)     # Sample in the null distribution
        random_dist[i] = random_corr

    r = rdm_corr(model, gt, flatten=False)  # True correlation obtained with the model
    p = np.sum(random_dist > r) / n         # Probability of true correlation under the null distribution

    return r, p, random_dist


def model_distribution(model_runs, gt):
    """
    Estimate the distribution of the model's correlation to the "true" model using different samples of brain activity
    for each stimulus obtained over several runs. Essentially, build model RDMs by sampling each stimulus from one of
    its runs, and each time compute the correlation to the "true" model.
    :param model_runs: RDM of the model for all stimuli over all runs
    :param gt: "True" RDM
    :return: Estimated distribution of model correlation to the "true" model
    """
    n = 10000
    random_dist = np.zeros(n)
    gt = flatten_matrix(gt)

    def sample_model():
        # Make a sample RDM by randomly selecting one of the dissimilarities between two categories across all runs
        # (i.e. each [category, category] pair in the RDM is taken from one of the pairs for those categories
        # across all runs)
        model_sample = []
        # For each [category, category] pair
        for i in range(n_cats):
            for j in range(i + 1, n_cats):
                distances = model_runs[i*n_runs:(i+1)*n_runs, j*n_runs:(j+1)*n_runs]    # Distances for all runs
                sample = np.random.choice(distances.flatten(), 1)[0]                    # Randomly select one
                model_sample.append(sample)                                             # Add to the RDM
        model_sample = np.array(model_sample, dtype=np.float64)
        return model_sample

    # For the desired number of samples to estimate the distribution
    for i in range(n):
        model_sample = sample_model()                               # Sample an RDM from the model
        corr_sample = rdm_corr(model_sample, gt, flatten=False)     # Correlation with the "true" RDM for this sample
        random_dist[i] = corr_sample

    return random_dist


if __name__ == '__main__':
    assert len(sys.argv) == 2
    subj = int(sys.argv[1])

    gt_rdm = np.load('data/gt_rdm.npy')

    # Metric mean significance p-tests (is the RDM using the distance metric a good model of the "true" RDM
    for metric in ['emd', 'euclidean', 'corrinv']:
        mean_rdm = np.load('results/s{:02d}/{}_rdm.npz'.format(subj, metric))['mean']       # Load the metric RDM
        r, p, dist = permutation_test(mean_rdm, gt_rdm)                                     # Permutation test
        np.savez('results/s{:02d}/{}_perm.npz'.format(subj, metric), r=r, p=p, dist=dist)   # Save results

    # Metric distributions (distribution of correlations between model RDMs and "true" RDM)
    dists = []
    for metric in ['emd', 'euclidean', 'corrinv']:
        rdm = np.load('results/s{:02d}/{}_rdm.npz'.format(subj, metric))['runs']    # Load the metric RDM for all runs
        dist = model_distribution(rdm, gt_rdm)                                      # Distribution of RDM correlations
        dists.append(dist)
        np.save('results/s{:02d}/{}_distribution.npy'.format(subj, metric), dist)   # Save the distribution

    # Metric distribution pairwise t-test comparisons (are the models different from each other)
    _, p_emd_euclidean = stats.ttest_ind(dists[0], dists[1], equal_var=False)   # Probability emd=euclidean model
    _, p_emd_corrinv = stats.ttest_ind(dists[0], dists[2], equal_var=False)     # Probability emd=(1-correlation) model
    np.savez('results/s{:02d}/comparisons.npz'.format(subj),                    # Save the comparison results
             p_emd_euclidean=p_emd_euclidean, p_emd_corrinv=p_emd_corrinv)
