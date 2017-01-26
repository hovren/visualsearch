import os

import h5py
import cv2

def sift_file_for_image(path):
    root, _ = os.path.splitext(path)
    sift_file = os.path.join(root + '.sift.h5')
    return sift_file


def load_descriptors_and_keypoints(path, *, descriptors=True, keypoints=True):
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


def keypoint_inside_roi(kp, roi):
    rx, ry, rw, rh = roi
    x, y = kp.pt
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)


def filter_roi(des, kps, roi):
    valid = [i for i, kp in enumerate(kps) if keypoint_inside_roi(kp, roi)]
    kps = [kp for i, kp in enumerate(kps) if i in valid]
    des = des[valid]
    return des, kps


def calculate_sift(image, roi=None):
    detector = cv2.xfeatures2d.SIFT_create()
    if roi is None: # Entire image
        kps, des = detector.detectAndCompute(image, None)
    else:
        kps = detector.detect()
        _, kps = filter_roi([], kps, roi)
        des = detector.compute(kps)

    return des, kps