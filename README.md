# RSA Earth Mover's Distance

Project to perform RSA analysis using the Earth Mover's Distance as the distance metric between patterns of activity
for stimuli. The hypothesis is that fMRI data does not consist of arbitrarily ordered vectors, and that it is rather
deeply related to the 3D structure of the brain. In other words, the spatial position of voxels relative to others
is important in determining what the patterns represent as a whole, and so a dissimilarity metric that considers the
underlying metric space (assumed here to be 3D distance in the brain) is beneficial.

## Generating RDMs
Running `python distance.py [subject number (1-8)]` will generate the RDMs over the dataset of images using 3 distance
metrics:
1. Earth Mover's Distance
2. Euclidean Distance
3. 1 - Pearson Correlation

For each distance metric, 2 RDMs are created:
1. RDMs between the mean patterns of activation for a stimulus across all runs
2. RDMs between the patterns of activation for every stimulus for every run

The RDMs are saved as `results/[subject number]/[distance]_rdm.npz`.

## Significance tests
Running `python analysis.py [subject number (1-8)]` will run significance tests for the RDMs compared to the "true" RDM 
using nonparametric permutation tests. In addition, a t-test is done between the distance metrics comparing the 
distributions of their correlations to the "true" RDM, in order to see if their differences were statistically 
significant.

Permutation test significance results for similarity between each distance metric RDM and the "true" RDM are saved as 
`results/[subject number]/[distance]_perm.npz`.

T-test significance results for the comparison between distance metric RDMs are saved as 
`results/[subject number]/comparisons.npz`.

## Plotting
`plotting.py` contains functions for plotting:
1. An RDM
2. A dendrogram for an RDM
3. A permutation test null distribution and result
4. A comparison between two empirical distributions
