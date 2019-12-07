import numpy as np
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

    r = rdm_corr(model, gt)
    p = np.sum(random_dist > r) / n

    return r, p, random_dist


def model_dist(model_runs, gt):
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
