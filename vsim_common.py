import collections
import os
import cv2

import h5py
import numpy as np
import scipy.spatial


def sift_file_for_image(image_path):
    assert image_path.endswith('.jpg')
    return os.path.splitext(image_path)[0] + '.sift.h5'


def load_SIFT_descriptors(path):
    with h5py.File(path, 'r') as f:
        return f['descriptors'].value

def load_SIFT_file(path):
    with h5py.File(path, 'r') as f:
        descriptors = f['descriptors'].value
        g = f['keypoints']
        points = g['pt'].value
        sizes = g['size'].value
        angles = g['angle'].value
        responses = g['response'].value
        octaves = g['octave'].value

        keypoints = []
        for (x, y), size, angle, response, octave in zip(points, sizes, angles, responses, octaves):
            kp = cv2.KeyPoint(x, y, size, angle, response, octave)
            keypoints.append(kp)

    return descriptors, keypoints

def save_vocabulary(means, path):
    with h5py.File(path, 'w') as f:
        f['vocabulary'] = means


def kmeans_score(data, centroids, labels):
    return sum(np.linalg.norm(x - centroids[l]) ** 2 for x, l in zip(data, labels))


def load_vocabulary(path):
    with h5py.File(path, 'r') as f:
        return f['vocabulary'].value


def label_data(centroids, data):
    X = scipy.spatial.distance.cdist(data, centroids, 'sqeuclidean')
    labels = np.argmin(X, axis=1)
    assert len(labels) == len(data)
    return labels

def descriptors_to_bow_vector(descriptors, vocabulary):
    labels = label_data(vocabulary, descriptors)
    document_word_count = collections.Counter(labels.flatten().tolist())
    document_word_freq = np.array([document_word_count[i] for i in range(len(vocabulary))])
    return document_word_freq