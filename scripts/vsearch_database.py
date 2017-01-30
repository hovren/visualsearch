#!/usr/bin/env python

import argparse
import collections
import glob
import multiprocessing
import os
import sys
from itertools import repeat

import annoy
import cv2
import h5py
import numpy as np
import tqdm

from vsearch.utils import load_descriptors_and_keypoints, FEATURE_TYPES, load_vocabulary


class AnnBowComputer:
    @staticmethod
    def build_index(vocabulary, feat_size):
        index = annoy.AnnoyIndex(feat_size, metric='euclidean')
        vocabulary_size = len(vocabulary)
        with tqdm.tqdm(total=len(vocabulary), desc="Building Index") as pbar:
            for i, x in enumerate(vocabulary):
                index.add_item(i, x)
                pbar.update(1)
        index.build(n_trees=10)

        return index, vocabulary_size

    def __init__(self, index_file, voc_size, feat_type):
        self.index_file = index_file
        self.feat_type = feat_type
        self.vocabulary_size = voc_size

    def compute(self, descriptors_file_path):
        index = annoy.AnnoyIndex(self.feat_type.featsize, metric='euclidean')
        index.load(self.index_file)
        descriptors, _ = load_descriptors_and_keypoints(descriptors_file_path, keypoints=False)
        key = os.path.basename(descriptors_file_path).split(self.feat_type.extension)[0]
        document_word_count = collections.Counter()
        for des in descriptors:
            res = index.get_nns_by_vector(des, 1) # Nearest neighbour
            label = res[0]
            document_word_count[label] += 1
        document_word_freq = np.array([document_word_count[i] for i in range(self.vocabulary_size)])
        return key, document_word_freq


def worker(args):
    source_file, computer = args
    return computer.compute(source_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "Create Bag of Words database from all files in directory using a vocabulary"
    parser.add_argument('directory')
    parser.add_argument('vocabulary')
    parser.add_argument('out')
    parser.add_argument('feature', choices=list(FEATURE_TYPES.keys()))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--nproc', type=int)
    args = parser.parse_args()
    
    directory = os.path.expanduser(args.directory)
    vocabulary_file = os.path.expanduser(args.vocabulary)
    out_file = os.path.expanduser(args.out)
    
    if os.path.exists(out_file) and not args.overwrite:
        print('{} already exists. Rerun with --overwrite'.format(out_file))
        sys.exit(-1)
        
    vocabulary = load_vocabulary(vocabulary_file)
    print('Vocabulary {} contained {} visual words'.format(vocabulary_file, len(vocabulary)))

    feat_type = FEATURE_TYPES[args.feature]
    print('Using {} features with {:d} dimensions'.format(feat_type.name, feat_type.featsize))

    if not vocabulary.shape[1] == feat_type.featsize:
        print("ERROR: Vocabulary word dimensionality ({:d}) did not match the selected feature type {} ({:d})".format(
            vocabulary.shape[1], feat_type.name, feat_type.featsize))
        sys.exit(-1)

    glob_expr = '*' + feat_type.extension
    source_files = glob.glob(os.path.join(directory, glob_expr))

    index, voc_size = AnnBowComputer.build_index(vocabulary, feat_type.featsize)
    index_path = '/tmp/bow_db_creator_annoy_index.ann'
    if os.path.exists(index_path):
        print('Removing', index_path)
        os.remove(index_path)
    index.save(index_path)
    computer = AnnBowComputer(index_path, voc_size, feat_type)

    print('{} has {:d} source files'.format(directory, len(source_files)))

    print('Using {} processes'.format("all available" if args.nproc is None else args.nproc))

    with h5py.File(out_file, 'w') as f, multiprocessing.Pool(processes=args.nproc) as pool, tqdm.tqdm(total=len(source_files)) as pbar:
        for key, document_word_freq in pool.imap_unordered(worker, zip(source_files, repeat(computer))):
            assert key not in f.keys()
            f[key] = document_word_freq
            pbar.update(1)

        # Store vocabulary in database file
        f['vocabulary'] = vocabulary

    print('Wrote database to', out_file)
