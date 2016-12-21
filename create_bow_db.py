#!/usr/bin/env python

import argparse
import multiprocessing
import glob
import os
import sys
from itertools import repeat

import h5py
import tqdm
import cv2

from vsim_common import load_vocabulary, load_SIFT_descriptors, descriptors_to_bow_vector

SIFT_GRID_RADIUS = 4

def grid_sift(image, radius):
    h, w, *_ = image.shape
    keypoints = [cv2.KeyPoint(x, y, radius) for x in range(radius, w - radius, 2*radius)
                                            for y in range(radius, h - radius, 2*radius)]
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.compute(image, keypoints)

def compute_bow_vector(args):
    sift_file_path, vocabulary = args
    descriptors = load_SIFT_descriptors(sift_file_path)
    document_word_freq = descriptors_to_bow_vector(descriptors, vocabulary)
    key = os.path.basename(sift_file_path).split('.sift.h5')[0]
    return key, document_word_freq

def compute_gridded_bow_vector(args):
    image_file_path, vocabulary = args
    image = cv2.imread(image_file_path)
    kps, descriptors = grid_sift(image, SIFT_GRID_RADIUS)
    document_word_freq = descriptors_to_bow_vector(descriptors, vocabulary)
    key = os.path.basename(image_file_path).split('.jpg')[0]
    return key, document_word_freq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('vocabulary')
    parser.add_argument('out')
    parser.add_argument('--gridded', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    directory = os.path.expanduser(args.directory)
    vocabulary_file = os.path.expanduser(args.vocabulary)
    out_file = os.path.expanduser(args.out)
    
    if os.path.exists(out_file) and not args.overwrite:
        print('{} already exists. Rerun with --overwrite'.format(out_file))
        sys.exit(-1)
        
    vocabulary = load_vocabulary(vocabulary_file)
    print('Vocabulary {} contained {} visual words'.format(vocabulary_file, len(vocabulary)))

    if args.gridded:
        print('Using gridded SIFT')
        source_files = glob.glob(os.path.join(directory, '*.jpg'))
        compute_func = compute_gridded_bow_vector
    else:
        print('Using SIFT keypoints')
        source_files = glob.glob(os.path.join(directory, '*.sift.h5'))
        compute_func = compute_bow_vector

    print('{} has {:d} source files'.format(directory, len(source_files)))

    with h5py.File(out_file, 'w') as f, multiprocessing.Pool() as pool, tqdm.tqdm(total=len(source_files)) as pbar:
        for key, document_word_freq in pool.imap_unordered(compute_func, zip(source_files, repeat(vocabulary))):
            assert key not in f.keys()
            f[key] = document_word_freq
            pbar.update(1)

        f['vocabulary'] = vocabulary

    print('Wrote database to', out_file)
