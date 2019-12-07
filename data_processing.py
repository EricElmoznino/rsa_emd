import os
import numpy as np
from nilearn import image


n_runs = 10

categories = ['monkey', 'lemur', 'duck', 'warbler', 'ladybug', 'moth']
n_cats = len(categories)


def read_subject(subj_num):
    data_dir = 'data/s{:02d}'.format(subj_num)

    brain_fname = os.path.join(data_dir, 'glm_T_stats_perrun.nii.gz')
    brain = image.load_img(brain_fname)
    brain = np.array(brain.dataobj)
    brain = brain.astype(np.float64)
    brain = brain.transpose(3, 0, 1, 2)
    brain_category_ordered = []
    for c in range(n_cats):
        for r in range(0, n_runs * n_cats, n_cats):
            brain_category_ordered.append(brain[c + r])
    brain = brain_category_ordered

    vt_mask_fname = os.path.join(data_dir, 'vt_mask.nii.gz')
    vt_mask = image.load_img(vt_mask_fname)
    vt_mask = np.array(vt_mask.dataobj)
    vt_mask = vt_mask.astype(np.bool)

    return brain, vt_mask
