import argparse
import glob
import os
import sys
import time
import multiprocessing
import itertools

import cv2
import numpy as np
import tqdm
import h5py
import scipy.io

from vsim_common import load_SIFT_file, save_keypoints_and_descriptors

COLOR_NAMES = ['black', 'blue', 'brown', 'grey', 'green', 'orange',
               'pink', 'purple', 'red', 'white', 'yellow']
COLOR_RGB = [[0, 0, 0] , [0, 0, 1], [.5, .4, .25] , [.5, .5, .5] , [0, 1, 0] , [1, .8, 0] ,
             [1, .5, 1] ,[1, 0, 1], [1, 0, 0], [1, 1, 1 ] , [ 1, 1, 0 ]]

COLORNAME_DESCRIPTOR_LENGTH = 11

def colornames_image(im, w2c, mode='index'):
    im = im.astype('double')
    idx = np.floor(im[..., 0] / 8) + 32 * np.floor(im[..., 1] / 8) + 32 * 32 * np.floor(im[..., 2] / 8)
    m = w2c[idx.astype('int')]

    if mode == 'index':
        return np.argmax(m, 2)
    elif mode == 'probability':
        return m
    else:
        raise ValueError("No such mode: '{}'".format(mode))


def root_for_sift_file(fname):
    return fname.split('.sift.h5')[0]


def cname_file_for_sift_file(fname):
    return root_for_sift_file(fname) + '.cname.h5'


def worker(args):
    froot, w2c = args
    sift_file = froot + '.sift.h5'
    sift_des, keypoints = load_SIFT_file(sift_file, descriptors=False)
    image_file = froot + '.jpg'
    outfile = cname_file_for_sift_file(sift_file)
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    cname_des = np.zeros((len(keypoints), COLORNAME_DESCRIPTOR_LENGTH), dtype='float32')
    for i, kp in enumerate(keypoints):
        r = kp.size # Mask/patch radius (note: kp.size is diameter, but we want twice the size anyway)
        R = int(np.ceil(r))

        # Cut out patch and extract colornames
        x, y = np.round(np.array(kp.pt)).astype('int')
        xmin = max(x-R, 0)
        xmax = min(x + R, W-1)
        ymin = max(y - R, 0)
        ymax = min(y + R, H-1)
        my, mx = np.mgrid[ymin:ymax+1, xmin:xmax+1]
        patch = image[my, mx]
        cname_patch = colornames_image(patch, w2c, mode='probability')

        # Histogram (normalized) of colornames probabilities within circular radius
        d = np.sqrt((x - mx)**2 + (y - my)**2)
        mask = (d <= r)
        probabilities = cname_patch[mask]
        cname_des[i] = np.sum(probabilities, 0) / len(probabilities)

    save_keypoints_and_descriptors(outfile, keypoints, cname_des)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()

    w2c = scipy.io.loadmat('w2c.mat')['w2c']

    directory = os.path.expanduser(args.dir)
    sift_files = glob.glob(os.path.join(directory, '*.sift.h5'))
    missing_cname_files = [root_for_sift_file(f) for f in sift_files if not os.path.exists(cname_file_for_sift_file(f))]
    print('{:d} of {:d} SIFT files in {} is missing colornames descriptors'.format(len(missing_cname_files), len(sift_files), directory))

    if not missing_cname_files:
        print('Nothing to do')
        sys.exit(0)

    with multiprocessing.Pool() as pool, tqdm.tqdm(total=len(missing_cname_files)) as pbar:
        for froot in pool.imap_unordered(worker, zip(missing_cname_files, itertools.repeat(w2c))):
            pbar.update(1)
