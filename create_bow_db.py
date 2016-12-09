#!/usr/bin/env python

import argparse
import multiprocessing
import glob
import os
import sys
from itertools import repeat

import h5py
import tqdm

from vsim_common import load_vocabulary, load_SIFT_descriptors, descriptors_to_bow_vector

def compute_bow_vector(args):
    path, vocabulary = args
    descriptors = load_SIFT_descriptors(path)
    document_word_freq = descriptors_to_bow_vector(descriptors, vocabulary)
    key = os.path.basename(path).split('.sift.h5')[0]
    return key, document_word_freq

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

    with h5py.File(out_file, 'w') as f, multiprocessing.Pool() as pool, tqdm.tqdm(total=len(sift_files)) as pbar:
        for key, document_word_freq in pool.imap_unordered(compute_bow_vector, zip(sift_files, repeat(vocabulary))):
            assert key not in f.keys()
            f[key] = document_word_freq
            pbar.update(1)

        f['vocabulary'] = vocabulary

    print('Wrote database to', out_file)
