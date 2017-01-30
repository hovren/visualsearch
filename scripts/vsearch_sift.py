#/usr/bin/env python3

import argparse
import os

import cv2
import tqdm

from vsearch.utils import save_keypoints_and_descriptors, find_images
from vsearch.sift import sift_file_for_image, calculate_sift


def find_missing(directory):
    return [path for path in find_images(directory) if not os.path.exists(sift_file_for_image(path))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    
    directory = os.path.expanduser(args.directory)
    
    detector = cv2.xfeatures2d.SIFT_create()
    
    missing = find_missing(directory)
    print('Calculate SIFT features for {:d} images'.format(len(missing)))

    for path in tqdm.tqdm(missing):
        img = cv2.imread(path)
        desc, kps = calculate_sift(img)
        outpath = sift_file_for_image(path)
        save_keypoints_and_descriptors(outpath, kps, desc)

    print('Done')            
