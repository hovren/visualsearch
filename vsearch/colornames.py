# Copyright 2017 Hannes Ovrén
#
# This file is part of vsearch.
#
# vsearch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# vsearch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with vsearch.  If not, see <http://www.gnu.org/licenses/>.

import os

import numpy as np
import scipy.io

from .sift import calculate_sift
from .utils import FEATURE_TYPES, filter_roi

CNAMES = FEATURE_TYPES['colornames']

COLOR_NAMES = ['black', 'blue', 'brown', 'grey', 'green', 'orange',
               'pink', 'purple', 'red', 'white', 'yellow']
COLOR_RGB = [[0, 0, 0] , [0, 0, 1], [.5, .4, .25] , [.5, .5, .5] , [0, 1, 0] , [1, .8, 0] ,
             [1, .5, 1] ,[1, 0, 1], [1, 0, 0], [1, 1, 1 ] , [ 1, 1, 0 ]]

COLORNAMES_TABLE_PATH = os.path.join(os.path.dirname(__file__), 'colornames_w2c.mat')
COLORNAMES_TABLE = scipy.io.loadmat(COLORNAMES_TABLE_PATH)['w2c']

def colornames_image(image, mode='index'):
    """Apply color names to an image

    Parameters
    --------------
    image : array_like
        The input image array (RxC)
    mode : str
        If 'index' then it returns an image where each element is the corresponding color name label.
        If 'probability', then the returned image has size RxCx11 where the last dimension are the probabilities for each
        color label.
        The corresponding human readable name of each label is found in the `COLOR_NAMES` list.

    Returns
    --------------
    Color names encoded image, as explained by the `mode` parameter.
    """
    image = image.astype('double')
    idx = np.floor(image[..., 0] / 8) + 32 * np.floor(image[..., 1] / 8) + 32 * 32 * np.floor(image[..., 2] / 8)
    m = COLORNAMES_TABLE[idx.astype('int')]

    if mode == 'index':
        return np.argmax(m, 2)
    elif mode == 'probability':
        return m
    else:
        raise ValueError("No such mode: '{}'".format(mode))


def cname_file_for_image(path):
    """Return color names descriptor filename corresponding to an image path"""
    root, _ = os.path.splitext(path)
    cname_file = os.path.join(root + '.cname.h5')
    return cname_file


def calculate_colornames(image, roi=None, keypoints=None):
    """Calculate colornames descriptors and/or keypoints

    For color names we still use SIFT to detect keypoints.
    For each keypoint we use a radius that is twice that of the corresponding SIFT feature,
    and calculate a histogram of color name probabilities within this region.
    The histogram is then normalized to not depend on the chosen radius.

    Parameters
    --------------
    image : array_like
        The input image array
    roi : array_like
        An optional region of interest encoded as (x, y, width, height)
        If None, then the whole image is used
    keypoints : list
        List of keypoints for which to compute the color names descriptors.
        If None, then SIFT keypoints for the image will be computed.

    Returns
    -------------
    descriptors :  array_like
        NxD array containing the N D-dimensional descriptor vectors
    keypoints : list
        List of N cv2.Keypoint objects
        """
    if not keypoints:
        _, keypoints = calculate_sift(image, only_keypoints=True)

    _, keypoints = filter_roi([], keypoints, roi)

    cname_des = np.zeros((len(keypoints), CNAMES.featsize), dtype='float32')
    H, W = image.shape[:2]
    for i, kp in enumerate(keypoints):
        r = kp.size  # Mask/patch radius (note: kp.size is diameter, but we want twice the size anyway)
        R = int(np.ceil(r))

        # Cut out patch and extract colornames
        x, y = np.round(np.array(kp.pt)).astype('int')
        xmin = max(x - R, 0)
        xmax = min(x + R, W - 1)
        ymin = max(y - R, 0)
        ymax = min(y + R, H - 1)
        my, mx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
        patch = image[my, mx]
        cname_patch = colornames_image(patch, mode='probability')

        # Histogram (normalized) of colornames probabilities within circular radius
        d = np.sqrt((x - mx) ** 2 + (y - my) ** 2)
        mask = (d <= r)
        probabilities = cname_patch[mask]
        cname_des[i] = np.sum(probabilities, 0) / len(probabilities)

    return cname_des, keypoints
