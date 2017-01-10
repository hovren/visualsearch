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

from vsim_common import load_vocabulary, load_SIFT_descriptors, descriptors_to_bow_vector, grid_sift, FEATURE_TYPES

SIFT_GRID_RADIUS = 4
SIFT_FEATURE_LENGTH = 128
EXACT_VOCABULARY_MAX_SIZE = 15000

class BowComputer:
    def __init__(self, vocabulary, feat_type):
        self.vocabulary = vocabulary
        self.feat_type = feat_type

    def compute(self, sift_file_path):
        descriptors = load_SIFT_descriptors(sift_file_path)
        document_word_freq = descriptors_to_bow_vector(descriptors, self.vocabulary)
        key = os.path.basename(sift_file_path).split(feat_type.extension)[0]
        return key, document_word_freq

class GridBowComputer:
    def __init__(self, vocabulary, radius, step):
        self.vocabulary = vocabulary
        self.radius = radius
        self.step = step

    def compute(self, image_file_path):
        image = cv2.imread(image_file_path)
        kps, descriptors = grid_sift(image, self.radius, self.step)
        document_word_freq = descriptors_to_bow_vector(descriptors, self.vocabulary)
        key = os.path.basename(image_file_path).split('.jpg')[0]
        return key, document_word_freq

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


    def compute(self, sift_file_path):
        #print('Loading', sift_file_path)
        #print('Loading index for', sift_file_path)
        index = annoy.AnnoyIndex(self.feat_type.featsize, metric='euclidean')
        index.load(self.index_file)
        #print('Index loaded for ', sift_file_path)
        descriptors = load_SIFT_descriptors(sift_file_path)
        key = os.path.basename(sift_file_path).split(self.feat_type.extension)[0]
        document_word_count = collections.Counter()
        for des in descriptors:
            #print('Running')
            res = index.get_nns_by_vector(des, 1) # Nearest neighbour
            #print('Res:', res)
            label = res[0]
            document_word_count[label] += 1
        document_word_freq = np.array([document_word_count[i] for i in range(self.vocabulary_size)])
        #print('Done with', sift_file_path)
        return key, document_word_freq


def worker(args):
    source_file, computer = args
    return computer.compute(source_file)


FEATURES_DICT = {ft.key: ft for ft in FEATURE_TYPES}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('vocabulary')
    parser.add_argument('out')
    parser.add_argument('feature', choices=list(FEATURES_DICT.keys()))
    parser.add_argument('--grid', nargs=2, type=int)
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

    feat_type = FEATURES_DICT[args.feature]
    print('Using {} features with {:d} dimensions'.format(feat_type.name, feat_type.featsize))

    if args.grid and feat_type.key == 'sift':
        radius, step = args.grid
        if step < 1:
            step = int(radius / 2)
        print('Using gridded SIFT with radius {:d} and step {:d}'.format(radius, step))
        source_files = glob.glob(os.path.join(directory, '*.jpg'))
        computer = GridBowComputer(vocabulary, radius, step)
    else:
        glob_expr = '*' + feat_type.extension
        source_files = glob.glob(os.path.join(directory, glob_expr))
        if len(vocabulary) > EXACT_VOCABULARY_MAX_SIZE:
            print('Large vocabulary ({:d} > {:d}), using approximate nearest neighbour'.format(len(vocabulary), EXACT_VOCABULARY_MAX_SIZE))

            index, voc_size = AnnBowComputer.build_index(vocabulary, feat_type.featsize)
            index_path = '/tmp/bow_db_creator_annoy_index.ann'
            if os.path.exists(index_path):
                print('Removing', index_path)
                os.remove(index_path)
            index.save(index_path)
            computer = AnnBowComputer(index_path, voc_size, feat_type)
        else:
            print('Small vocabulary, using exact kmeans')
            computer = BowComputer(vocabulary, feat_type)

    print('{} has {:d} source files'.format(directory, len(source_files)))

    print('Using {} processes'.format(args.nproc))

    with h5py.File(out_file, 'w') as f, multiprocessing.Pool(processes=args.nproc) as pool, tqdm.tqdm(total=len(source_files)) as pbar:
        for key, document_word_freq in pool.imap_unordered(worker, zip(source_files, repeat(computer))):
            assert key not in f.keys()
            f[key] = document_word_freq
            pbar.update(1)

        f['vocabulary'] = vocabulary

        if args.grid:
            f['grid_radius'] = radius
            f['grid_step'] = step

    print('Wrote database to', out_file)
