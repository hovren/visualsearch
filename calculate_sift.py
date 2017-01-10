#/usr/bin/env python3

import argparse
import os

import cv2
import tqdm

from vsim_common import sift_file_for_image, save_keypoints_and_descriptors


def find_missing(directory):
    def match(filename):
        if filename.endswith('.jpg'):
            #root = os.path.splitext(filename)[0]
            #sift_file = os.path.join(directory, root + '.sift.h5') 
            path = os.path.join(directory, filename)
            sift_file = sift_file_for_image(path)
            return not os.path.exists(sift_file)
        else:
            return False            
        
    return [os.path.join(directory, f) for f in os.listdir(directory) if match(f)]


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
        kps, desc = detector.detectAndCompute(img, None)
        outpath = sift_file_for_image(path)
        save_keypoints_and_descriptors(outpath, kps, desc)

    print('Done')            
