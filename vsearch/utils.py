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
import collections

import cv2
import h5py
import numpy as np


FeatureType = collections.namedtuple('FeatureType', 'key name extension featsize')

FEATURE_TYPES = {
    'sift': FeatureType('sift', 'SIFT', '.sift.h5', 128),
    'colornames': FeatureType('colornames', 'colornames', '.cname.h5', 11)
}

SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.png')


def load_descriptors_and_keypoints(path, *, descriptors=True, keypoints=True):
    """Load a descriptor/keypoint file

    Parameters
    -------------
    path : str
        Path to descriptor file
    descriptors : bool
        Whether to load the descriptors
    keypoints : bool
        Whether to load the keypoints

    Returns
    ------------------
    descriptors : array_like
        NxD array of N descriptor vectors, or None if asked to not return any
    keypoints : list
        List of N cv2.Keypoint objects, or None if asked to not return any
    """
    with h5py.File(path, 'r') as f:
        if descriptors:
            descriptors = f['descriptors'].value
        else:
            descriptors = None

        keypoint_list = []
        if keypoints:
            g = f['keypoints']
            points = g['pt'].value
            sizes = g['size'].value
            angles = g['angle'].value
            responses = g['response'].value
            octaves = g['octave'].value

            for (x, y), size, angle, response, octave in zip(points, sizes, angles, responses, octaves):
                kp = cv2.KeyPoint(x, y, size, angle, response, octave)
                keypoint_list.append(kp)

    return descriptors, keypoint_list


def save_keypoints_and_descriptors(path, kps, desc):
    """Save keypoints and descriptors to a HDF5 file"""
    with h5py.File(path, 'w') as f:
        f['descriptors'] = np.vstack(desc)
        g = f.create_group('keypoints')
        g['pt'] = np.vstack([kp.pt for kp in kps])
        g['size'] = np.vstack([kp.size for kp in kps])
        g['angle'] = np.vstack([kp.angle for kp in kps])
        g['response'] = np.vstack([kp.response for kp in kps])
        g['octave'] = np.vstack([kp.octave for kp in kps])


def find_images(directory):
    """Return list of full paths to images of a supported format in a directory"""
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.splitext(f)[-1] in SUPPORTED_IMAGE_EXTENSIONS]


def keypoint_inside_roi(kp, roi):
    """Check whether a keypoint is within the region of interest

    Parameters
    -----------------
    kp : cv2.Keypoint
        Keypoint
    roi : array_like
        An optional region of interest encoded as (x, y, width, height)

    Returns:
        True if keypoint.pt is within the ROI
    """
    rx, ry, rw, rh = roi
    x, y = kp.pt
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)


def filter_roi(des, kps, roi):
    """Filter a list of keypoints and descriptors to only those within the region of interest

    Parameters
    -------------------

    """
    valid = [i for i, kp in enumerate(kps) if keypoint_inside_roi(kp, roi)]
    kps = [kp for i, kp in enumerate(kps) if i in valid]
    des = des[valid]
    return des, kps


def save_vocabulary(means, path):
    with h5py.File(path, 'w') as f:
        f['vocabulary'] = means


def load_vocabulary(path):
    with h5py.File(path, 'r') as f:
        return f['vocabulary'].value


def image_for_descriptor_file(desc_path):
    try:
        feat_type = [ft for ft in FEATURE_TYPES.values() if desc_path.endswith(ft.extension)][0]
    except KeyError:
        raise ValueError("{} does not match any known feature type extension".format(os.path.basename(desc_path)))

    fname = os.path.basename(desc_path)
    root = fname[:-len(feat_type.extension)]
    directory = os.path.dirname(desc_path)

    images = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        path = os.path.join(directory, root + ext)
        if os.path.exists(path):
            images.append(path)

    if not images:
        raise ValueError("No matching image found")
    elif len(images) == 1:
        return images[0]
    else:
        raise ValueError("Multiple matching files found")
