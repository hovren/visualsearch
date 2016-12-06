#!/usr/bin/env python3

import argparse
import glob
import os
import time

import cv2
import h5py
import numpy as np

import tqdm

def load_SIFT_descriptors(path):
    with h5py.File(path, 'r') as f:
        return f['descriptors'].value
        
def save_vocabulary(means, path):
    with h5py.File(path, 'w') as f:
        f['vocabulary'] = means  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('out')
    parser.add_argument('size', type=int)
    args = parser.parse_args()
    
    out_path = os.path.expanduser(args.out)
    sift_files = glob.glob(os.path.join(os.path.expanduser(args.directory), '*.sift.h5'))[:2]

    sift_descriptors = []
    for path in tqdm.tqdm(sift_files):
        desc = load_SIFT_descriptors(path)
        sift_descriptors.append(desc)
     
    data = np.vstack(sift_descriptors)
    print('Loaded {} SIFT descriptors from {} files'.format(len(data), len(sift_descriptors)))
    
    max_iter = 5
    eps = 1.0 # Not used
    termcrit = (cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    attempts = 10
    clusters = args.size
    print('Clustering vocabulary with K={}, {:d} iterations and {:d} attempts'.format(args.size, max_iter, attempts))
    
    t0 = time.time()
    score, labels, means = cv2.kmeans(data, clusters, None, termcrit, attempts, cv2.KMEANS_RANDOM_CENTERS)
    elapsed = time.time() - t0
    
    print('Clustering took {:.1f} seconds'.format(elapsed))
    print('K-means compactness score {:.2f}'.format(score))
    out_file = os.path.expanduser(args.out)
    save_vocabulary(means, out_file)
    print('Saved vocabulary to', out_file)
