import os

import cv2

from vsearch.utils import filter_roi, FEATURE_TYPES

SIFT = FEATURE_TYPES['sift']

def sift_file_for_image(path):
    root, _ = os.path.splitext(path)
    sift_file = os.path.join(root + SIFT.extension)
    return sift_file


def calculate_sift(image, roi=None, only_keypoints=False):
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