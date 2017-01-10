#!/usr/bin/env python3

import argparse
import glob
import os
import time
import heapq

import h5py
import numpy as np
import scipy.cluster
import tqdm

from vsim_common import load_SIFT_descriptors, kmeans_score, save_vocabulary, FEATURE_TYPES

# KMeans providers
kmeans_providers = ['scipy'] # Scipy should always be installed
try:
    import cv2
    kmeans_providers.append('opencv')
except ImportError:
    pass

try:
    import sklearn.cluster
    kmeans_providers.append('sklearn')
except ImportError:
    pass

try:
    from annoy_kmeans import approx_kmeans_annoy
    kmeans_providers.append('annoy')
except ImportError:
    pass


FEATURES_DICT = {ft.key: ft for ft in FEATURE_TYPES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('out')
    parser.add_argument('size', type=int)
    parser.add_argument('feature', choices=list(FEATURES_DICT.keys()))
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--tries', type=int, default=1)
    parser.add_argument('--nprocs', type=int, default=1)
    parser.add_argument('--kmeans', choices=kmeans_providers, default='scipy')
    args = parser.parse_args()
    
    out_path = os.path.expanduser(args.out)
    feat_type = FEATURES_DICT[args.feature]
    glob_expr = '*' + feat_type.extension
    descriptor_files = glob.glob(os.path.join(os.path.expanduser(args.directory), glob_expr))

    descriptors = []
    print('Loading {} descriptors...'.format(args.feature))
    for path in tqdm.tqdm(descriptor_files):
        desc = load_SIFT_descriptors(path) # Note: All descriptors are loaded by the same function
        descriptors.extend(desc)

    print('Feature dimensions:', len(descriptors[0]))

    #data = np.vstack(sift_descriptors)
    #data = sift_descriptors
    print('Loaded {} {} descriptors from {} files'.format(len(descriptors), args.feature, len(descriptors)))
    
    iterations = args.iterations
    attempts = args.tries
    clusters = args.size

    if args.kmeans in ['annoy']:
        data = descriptors
    else:
        data = np.vstack(descriptors)
    
    
    print('Clustering vocabulary using {} with K={}, {:d} iterations and {:d} attempts'.format(args.kmeans, args.size, iterations, attempts))

    t0 = time.time()
    if args.kmeans == 'opencv':
        eps = 1.0 # Not used
        termcrit = (cv2.TERM_CRITERIA_MAX_ITER, iterations, eps)
        score, labels, centroids = cv2.kmeans(data, clusters, None, termcrit, attempts, cv2.KMEANS_RANDOM_CENTERS)
        print('Scipy score:', kmeans_score(data, centroids, labels))

    elif args.kmeans == 'scipy':
        score = np.inf
        centroids = None
        for i in tqdm.tqdm(range(attempts)):
            iter_centroids, labels = scipy.cluster.vq.kmeans2(data, clusters, iterations, minit='points')
            iter_score = kmeans_score(data, iter_centroids, labels)
            if iter_score < score:
                centroids = iter_centroids
                score = iter_score

    elif args.kmeans == 'sklearn':
        #kmeans = sklearn.cluster.KMeans(clusters, init='random', n_init=attempts, max_iter=iterations, n_jobs=-1, copy_x=False)
        kmeans = sklearn.cluster.MiniBatchKMeans(clusters, init='random', batch_size=100, n_init=attempts, max_iter=iterations, compute_labels=False, verbose=True)
        kmeans.fit(data)
        try:
            score = kmeans.inertia_
            labels = kmeans.labels_
        except AttributeError:
            score = -1
            labels = []
        centroids = kmeans.cluster_centers_

    elif args.kmeans == 'annoy':
        score = np.inf
        labels = None
        centroids = None
        with tqdm.tqdm(total=iterations*attempts) as pbar:
            def cb(score, num_empty):
                pbar.update(1)
            centroids, labels, score = approx_kmeans_annoy(data, clusters, iterations, iter_callback=cb, nprocs=args.nprocs)
                
    elapsed = time.time() - t0
    
    print('Clustering took {:.1f} seconds'.format(elapsed))
    print('K-means compactness score {:g}'.format(score))
    out_file = os.path.expanduser(args.out)
    save_vocabulary(centroids, out_file)
    print('Saved vocabulary to', out_file)
