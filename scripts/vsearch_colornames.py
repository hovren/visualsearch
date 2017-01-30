import argparse
import os
import multiprocessing

import cv2
import tqdm

from vsearch.utils import load_descriptors_and_keypoints, FEATURE_TYPES, find_images, save_keypoints_and_descriptors
from vsearch.colornames import cname_file_for_image, calculate_colornames
from vsearch.sift import sift_file_for_image

SIFT = FEATURE_TYPES['sift']
CNAME = FEATURE_TYPES['colornames']


def worker(image_path):
    cname_file = cname_file_for_image(image_path)
    sift_file = sift_file_for_image(image_path)

    try:
        sift_des, keypoints = load_descriptors_and_keypoints(sift_file, descriptors=False)
    except IOError:
        keypoints = None

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    descriptors, keypoints = calculate_colornames(image, keypoints=keypoints)

    save_keypoints_and_descriptors(cname_file, keypoints, descriptors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()

    directory = os.path.expanduser(args.dir)
    missing = []
    for f in find_images(directory):
        if not os.path.exists(cname_file_for_image(f)):
            missing.append(f)

    print('{:d} files in {} is missing colornames descriptors'.format(len(missing), directory))

    with multiprocessing.Pool() as pool, tqdm.tqdm(total=len(missing)) as pbar:
        for _ in pool.imap_unordered(worker, missing):
            pbar.update(1)
