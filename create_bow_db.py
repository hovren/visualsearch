#!/usr/bin/env python

import argparse
import collections
import glob
import os
import sys

import h5py
import numpy as np
import scipy.spatial
import tqdm

from vsim_common import load_vocabulary, load_SIFT_descriptors, descriptors_to_bow_vector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('vocabulary')
    parser.add_argument('out')
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

    sift_files = glob.glob(os.path.join(directory, '*.sift.h5'))
    print('{} has {:d} SIFT files'.format(directory, len(sift_files)))

    with h5py.File(out_file, 'w') as f:
        for path in tqdm.tqdm(sift_files):
            descriptors = load_SIFT_descriptors(path)
            document_word_freq = descriptors_to_bow_vector(descriptors, vocabulary)
            key = os.path.basename(path).split('.sift.h5')[0]
            assert key not in f.keys()
            f[key] = document_word_freq

    print('Wrote database to', out_file)
