import collections
import os
import cv2

import h5py
import numpy as np
import scipy.spatial

import matplotlib.pyplot as plt

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


def inside_roi(kp, roi):
    rx, ry, rw, rh = roi
    x, y = kp.pt
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)


def grid_sift(image, radius, step):
    h, w, *_ = image.shape
    keypoints = [cv2.KeyPoint(x, y, radius*2) for x in range(radius, w - radius, step)
                                              for y in range(radius, h - radius, step)]
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.compute(image, keypoints)

class VisualDatabase:
    def __init__(self, image_dict, vocabulary, stop_bottom=0, stop_top=0):
        self.vocabulary = vocabulary
        self._build_db(image_dict)

        if stop_top or stop_bottom:
            self._apply_stop_list(stop_bottom, stop_top)

    def _build_db(self, image_dict):
        self._image_dict = image_dict

        # Calculate inverse document frequency (IDF)
        # The number of documents that has at least one occurence of a word
        database_word_count = None
        database_total_word_count = None
        for v in image_dict.values():
            if database_word_count is None:
                database_word_count = np.ones_like(v)
                database_total_word_count = np.ones_like(v)
            database_word_count += (v > 0)
            database_total_word_count += v
        self.__total_word_count = database_total_word_count

        self._log_idf = np.log(len(image_dict) / database_word_count)

        # Store TF-IDF vectors for all images
        self._image_words = {}
        for key, v in image_dict.items():
            tf = v.astype('float64') / np.sum(v)
            tfidf = tf * self._log_idf
            self._image_words[key] = tfidf / np.linalg.norm(tfidf)

    def _descriptor_to_vector(self, des):
        return descriptors_to_bow_vector(des, self.vocabulary)

    def _apply_stop_list(self, stop_bottom, stop_top):
        word_count_order = np.argsort(self.__total_word_count)
        N_bottom = int(np.round(len(word_count_order) * stop_bottom))
        N_top = int(np.round(len(word_count_order) * stop_top))
        print('Removing {:d} least and {:d} most occuring words in database'.format(N_bottom, N_top))

        voc_size = len(self.vocabulary)
        valid_idxs = word_count_order[np.arange(N_bottom, voc_size - N_top)]

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(self.__total_word_count[word_count_order])
        ax2.plot(self.__total_word_count[valid_idxs])

        self.vocabulary = self.vocabulary[valid_idxs]
        self._log_idf = self._log_idf[valid_idxs]

        for key, v in self._image_words.items():
            self._image_words[key] = v[valid_idxs]

        for key, v in self._image_dict.items():
            self._image_dict[key] = v[valid_idxs]

    def query_vector(self, Vq, method='default'):
        Vq = Vq.astype('float64')
        # tf = (Vq.astype('float64') / np.sum(Vq))
        if method == 'default':
            tf = Vq
        elif method == 'max':
            tf = 0.5 + 0.5 * Vq / Vq.max()
        else:
            raise ValueError("Uknown method '{}'".format(method))
        Vq_tfidf = tf * self._log_idf
        Vq_tfidf /= np.linalg.norm(Vq_tfidf)
        scores = [(key, np.dot(Vq_tfidf, Vdb)) for key, Vdb in self._image_words.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def query_image(self, image, roi, method='default', sift_file=None):
        if sift_file:
            print('Loading SIFT features from', sift_file)
            des, kps = load_SIFT_file(sift_file)
        else:
            print('Detecting and calculating SIFT descriptors')
            detector = cv2.xfeatures2d.SIFT_create()
            kps, des = detector.detectAndCompute(image, None)

        print('Found {} SIFT keypoints'.format(len(kps)))

        valid = [i for i, kp in enumerate(kps) if inside_roi(kp, roi)]
        kps = [kp for i, kp in enumerate(kps) if i in valid]
        des = des[valid]

        print('Features in ROI:', len(kps))
        print('Calculating BOW vector')
        Vq = self._descriptor_to_vector(des)

        return self.query_vector(Vq, method=method)

    @classmethod
    def from_file(cls, db_path, **kwargs):
        with h5py.File(db_path, 'r') as f:
            db_dict = {key: f[key].value for key in f if not key == 'vocabulary'}
            vocabulary = f['vocabulary'].value

        instance = cls(db_dict, vocabulary, **kwargs)
        return instance


class MultiVisualDatabase(VisualDatabase):
    def __init__(self, databases):
        self.databases = databases

        image_dict = self._build_image_dict()
        self._build_db(image_dict)

    def _descriptor_to_vector(self, des):
        vectors = [descriptors_to_bow_vector(des, db.vocabulary) for db in self.databases]
        return np.hstack(vectors)

    def _build_image_dict(self):
        # Get list of images
        db0 = self.databases[0]
        keys = db0._image_words.keys()

        image_dict = {}
        for key in keys:
            Vdb = np.hstack([db._image_dict[key] for db in self.databases])
            assert Vdb.ndim == 1
            image_dict[key] = Vdb
        return image_dict