#!/usr/bin/env python3

import argparse
import glob
import os
import time

import cv2
import h5py
import numpy as np
import scipy.cluster
import sklearn.cluster

import tqdm

from vsim_common import load_SIFT_descriptors, kmeans_score, save_vocabulary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('out')
    parser.add_argument('size', type=int)
    parser.add_argument('--kmeans', choices=['opencv', 'scipy', 'sklearn'], default='scipy')
    args = parser.parse_args()
    
    out_path = os.path.expanduser(args.out)
    sift_files = glob.glob(os.path.join(os.path.expanduser(args.directory), '*.sift.h5'))

    sift_descriptors = []
    print('Loading SIFT descriptors...')
    for path in tqdm.tqdm(sift_files):
        desc = load_SIFT_descriptors(path)
        sift_descriptors.append(desc)
     
    data = np.vstack(sift_descriptors)
    print('Loaded {} SIFT descriptors from {} files'.format(len(data), len(sift_descriptors)))
    
    iterations = 5
    attempts = 10
    clusters = args.size
    
    
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
        kmeans = sklearn.cluster.KMeans(clusters, init='random', n_init=attempts, max_iter=iterations, n_jobs=-1, copy_x=False)
        kmeans.fit(data)
        score = kmeans.inertia_
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
                
    elapsed = time.time() - t0
    
    print('Clustering took {:.1f} seconds'.format(elapsed))
    print('K-means compactness score {:g}'.format(score))
    out_file = os.path.expanduser(args.out)
    save_vocabulary(centroids, out_file)
    print('Saved vocabulary to', out_file)
