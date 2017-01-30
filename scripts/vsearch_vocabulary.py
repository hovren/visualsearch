#!/usr/bin/env python3

import argparse
import glob
import os
import sys
import time

import numpy as np
import sklearn.cluster
import tqdm

from vsearch.utils import load_descriptors_and_keypoints, save_vocabulary, FEATURE_TYPES


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "Create a vocabulary from image directory with computed features"
    parser.add_argument('directory')
    parser.add_argument('out')
    parser.add_argument('size', type=int)
    parser.add_argument('feature', choices=list(FEATURE_TYPES.keys()))
    parser.add_argument('--iterations', type=int, default=15)
    parser.add_argument('--tries', type=int, default=10)
    args = parser.parse_args()
    
    out_path = os.path.expanduser(args.out)
    feat_type = FEATURE_TYPES[args.feature]
    glob_expr = '*' + feat_type.extension
    descriptor_files = glob.glob(os.path.join(os.path.expanduser(args.directory), glob_expr))
    print(descriptor_files)

    descriptors = []
    print('Loading {} descriptors...'.format(feat_type.name))
    for path in tqdm.tqdm(descriptor_files):
        desc, _ = load_descriptors_and_keypoints(path, keypoints=False)
        if not desc.shape[1] == feat_type.featsize:
            print('ERROR: {} did not contain {} descriptors'.format(path, feat_type.name))
            print(desc.shape, feat_type)
            sys.exit(-1)
        descriptors.extend(desc)

    data = np.vstack(descriptors)

    print('Feature dimensions:', len(descriptors[0]))
    print('Loaded {} {} descriptors from {} files'.format(len(descriptors), feat_type.name, len(descriptors)))
    
    iterations = args.iterations
    attempts = args.tries
    clusters = args.size

    print('Clustering vocabulary with K={}, {:d} iterations and {:d} attempts'.format(args.size, iterations, attempts))

    t0 = time.time()
    kmeans = sklearn.cluster.MiniBatchKMeans(clusters, init='random', batch_size=100, n_init=attempts, max_iter=iterations, compute_labels=False, verbose=True)
    kmeans.fit(data)
    try:
        score = kmeans.inertia_
        labels = kmeans.labels_
    except AttributeError:
        score = -1
        labels = []
    centroids = kmeans.cluster_centers_
    elapsed = time.time() - t0
    
    print('Clustering took {:.1f} seconds'.format(elapsed))
    print('K-means compactness score {:g}'.format(score))
    out_file = os.path.expanduser(args.out)
    save_vocabulary(centroids, out_file)
    print('Saved vocabulary to', out_file)
