import os
import numpy as np
from nilearn import image


n_runs = 10     # Number of times each stimulus was shown

categories = ['monkey', 'lemur', 'duck', 'warbler', 'ladybug', 'moth']      # Stimuli
n_cats = len(categories)    # Number of stimuli


def read_subject(subj_num):
    """
    Reads fMRI data for a given subject and returns it as numpy arrays.
    :param subj_num: int (1-8) of the subject for which we want to read fMRI data
    :return: tuple(ndarray, ndarray) with the fMRI data (ordered by stimulus) and the ventral temporal mask
    """
    data_dir = 'data/s{:02d}'.format(subj_num)

    # Read brain voxels across all stimuli and all runs
    brain_fname = os.path.join(data_dir, 'glm_T_stats_perrun.nii.gz')
    brain = image.load_img(brain_fname)
    brain = np.array(brain.dataobj)         # Convert to numpy format
    brain = brain.astype(np.float64)
    brain = brain.transpose(3, 0, 1, 2)     # "Time" axis first (stimulus/run)
    brain_category_ordered = []
    for c in range(n_cats):                 # Reorder the data by stimulus instead of by run.
        for r in range(0, n_runs * n_cats, n_cats):
            brain_category_ordered.append(brain[c + r])
    brain = brain_category_ordered

    # Read ventral temporal mask
    vt_mask_fname = os.path.join(data_dir, 'vt_mask.nii.gz')
    vt_mask = image.load_img(vt_mask_fname)
    vt_mask = np.array(vt_mask.dataobj)     # Convert to numpy format
    vt_mask = vt_mask.astype(np.bool)

    return brain, vt_mask
