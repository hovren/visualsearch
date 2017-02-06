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

import cv2

from vsearch.utils import filter_roi, FEATURE_TYPES

SIFT = FEATURE_TYPES['sift']


def sift_file_for_image(path):
    """Return SIFT descriptor filename corresponding to an image path"""
    root, _ = os.path.splitext(path)
    sift_file = os.path.join(root + SIFT.extension)
    return sift_file


def calculate_sift(image, roi=None, only_keypoints=False):
    """Calculate SIFT descriptors and/or keypoints

    Parameters
    --------------
    image : array_like
        The input image array
    roi : array_like
        An optional region of interest encoded as (x, y, width, height)
        If None, then the whole image is used
    only_keypoints : bool
        If True then only keypoints will be computed

    Returns
    -------------
    descriptors :  array_like
        NxD array containing the N D-dimensional descriptor vectors
    keypoints : list
        List of N cv2.Keypoint objects
    """
    detector = cv2.xfeatures2d.SIFT_create()
    if roi is None: # Entire image
        if only_keypoints:
            kps = detector.detect(image)
            des = None
        else:
            kps, des = detector.detectAndCompute(image, None)
    else:
        kps = detector.detect()
        _, kps = filter_roi([], kps, roi)
        if not only_keypoints:
            des = detector.compute(kps)
        else:
            des = None

    return des, kps