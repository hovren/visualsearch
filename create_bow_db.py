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

from vsim_common import load_vocabulary, load_SIFT_descriptors, descriptors_to_bow_vector, grid_sift

SIFT_GRID_RADIUS = 4

class BowComputer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def compute(self, sift_file_path):
        descriptors = load_SIFT_descriptors(sift_file_path)
        document_word_freq = descriptors_to_bow_vector(descriptors, self.vocabulary)
        key = os.path.basename(sift_file_path).split('.sift.h5')[0]
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


def worker(args):
    source_file, computer = args
    return computer.compute(source_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('vocabulary')
    parser.add_argument('out')
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

    if args.grid:
        radius, step = args.grid
        if step < 1:
            step = int(radius / 2)
        print('Using gridded SIFT with radius {:d} and step {:d}'.format(radius, step))
        source_files = glob.glob(os.path.join(directory, '*.jpg'))
        computer = GridBowComputer(vocabulary, radius, step)
    else:
        print('Using SIFT keypoints')
        source_files = glob.glob(os.path.join(directory, '*.sift.h5'))
        computer = BowComputer(vocabulary)

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
