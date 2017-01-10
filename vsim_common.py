import collections
import os
import cv2

import h5py
import numpy as np
import scipy.spatial
import annoy

import matplotlib.pyplot as plt

FeatureType = collections.namedtuple('FeatureType', 'key name extension featsize')

FEATURE_TYPES = (
    FeatureType('sift', 'SIFT', '.sift.h5', 128),
    FeatureType('colornames', 'colornames', '.cname.h5', 11)
)


def sift_file_for_image(image_path):
    assert image_path.endswith('.jpg')
    return os.path.splitext(image_path)[0] + '.sift.h5'


def load_SIFT_descriptors(path):
    with h5py.File(path, 'r') as f:
        return f['descriptors'].value

def load_SIFT_file(path, descriptors=True, keypoints=True):
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


def filter_roi(kps, des, roi):
    valid = [i for i, kp in enumerate(kps) if inside_roi(kp, roi)]
    kps = [kp for i, kp in enumerate(kps) if i in valid]
    des = des[valid]
    return kps, des


def grid_sift(image, radius, step):
    h, w, *_ = image.shape
    keypoints = [cv2.KeyPoint(x, y, radius*2) for x in range(radius, w - radius, step)
                                              for y in range(radius, h - radius, step)]
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.compute(image, keypoints)


def l1_distance(x, y):
    x = x / np.linalg.norm(x, ord=1)
    y = y / np.linalg.norm(y, ord=1)
    return np.linalg.norm(x - y, ord=1)


def cos_distance(x, y):
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    return 1 - np.dot(x, y)


class VisualDatabase:
    def __init__(self, image_dict, vocabulary, stop_bottom=0, stop_top=0):
        self.vocabulary = vocabulary
        self._build_db(image_dict)

        self._apply_stop_list(stop_bottom, stop_top)

        self._gridded = {}

    def _build_db(self, image_dict):
        self._image_dict = image_dict

        # Calculate inverse document frequency (IDF)
        # The number of documents that has at least one occurence of a word
        database_word_count = None
        database_total_word_count = None
        for v in image_dict.values():
            if database_word_count is None:
                # Start at 1 to make it robust when dividing later
                database_word_count = np.ones_like(v)
                database_total_word_count = np.ones_like(v)
            database_word_count += (v > 0)
            database_total_word_count += v
        self.__total_word_count = database_total_word_count

        self._log_idf = np.log(len(image_dict) / database_word_count)

        # Store TF-IDF vectors for all images
        self._tfidf_vectors = {}
        for key, v in image_dict.items():
            tf = v.astype('float64') #/ np.sum(v)
            tfidf = tf * self._log_idf
            self._tfidf_vectors[key] = tfidf

    def _descriptor_to_vector(self, des):
        return descriptors_to_bow_vector(des, self.vocabulary)

    def _apply_stop_list(self, stop_bottom, stop_top):
        word_count_order = np.argsort(self.__total_word_count)
        N_bottom = int(np.round(len(word_count_order) * stop_bottom))
        N_top = int(np.round(len(word_count_order) * stop_top))
        voc_size = len(self.vocabulary)
        valid_idxs = word_count_order[np.arange(N_bottom, voc_size - N_top)]
        self.valid_mask = np.zeros(voc_size, dtype='bool')
        self.valid_mask[valid_idxs] = 1
        self.stop_list = np.flatnonzero(~self.valid_mask)

    def query_vector(self, Vq, method='default', distance='cos', use_stop_list=True):
        Vq = Vq.astype('float64')
        # tf = (Vq.astype('float64') / np.sum(Vq))
        if method == 'default':
            tf = Vq
        elif method == 'max':
            tf = 0.5 + 0.5 * Vq / Vq.max()
        else:
            raise ValueError("Uknown method '{}'".format(method))

        Vq_tfidf = tf * self._log_idf

        if distance == 'cos':
            distance_func = cos_distance
        elif distance == 'l1':
            distance_func = l1_distance
        else:
            raise ValueError("Unknown distance '{}'".format(distance))

        if use_stop_list:
            print('Using stop list with {:d} words of {:d} total'.format(np.count_nonzero(~self.valid_mask), len(self.valid_mask)))
            scores = [(key, distance_func(Vq_tfidf[self.valid_mask], Vdb[self.valid_mask])) for key, Vdb in self._tfidf_vectors.items()]
        else:
            scores = [(key, distance_func(Vq_tfidf, Vdb)) for key, Vdb in self._tfidf_vectors.items()]
        scores.sort(key=lambda x: x[1]) #, reverse=True)
        return scores

    def query_image(self, image, roi, method='default', distance='cos', use_stop_list=True, sift_file=None,
                    grid=False, upsample=False, cname_file=None):
        if grid:
            if id(image) in self._gridded:
                print('Loading previous gridded descriptors')
                kps, des = self._gridded[id(image)]
            else:
                print('Calculating gridded SIFT descriptors')
                radius = 4
                kps, des = grid_sift(image, radius)
                self._gridded[id(image)] = (kps, des)
        elif sift_file or cname_file:
            if sift_file and cname_file:
                raise ValueError("Specify either sift_file or cname_file, not both")

            source = sift_file or cname_file
            print('Loading features from', source)
            des, kps = load_SIFT_file(source)
        else:
            print('Detecting and calculating SIFT descriptors')
            detector = cv2.xfeatures2d.SIFT_create()
            if upsample:
                x, y, w, h = roi
                query_image = image[y:y+h, x:x+w]
                query_image = cv2.resize(query_image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                print('Resized query image (correctly) from {} to {}'.format((h, w), query_image.shape))
                roi = (0, 0, query_image.shape[1], query_image.shape[0])
            else:
                query_image = image
            kps, des = detector.detectAndCompute(query_image, None)

        print('Found {} SIFT keypoints'.format(len(kps)))

        valid = [i for i, kp in enumerate(kps) if inside_roi(kp, roi)]
        kps = [kp for i, kp in enumerate(kps) if i in valid]
        des = des[valid]

        print('Features in ROI:', len(kps))
        print('Calculating BOW vector')
        Vq = self._descriptor_to_vector(des)

        return self.query_vector(Vq, method=method, distance=distance, use_stop_list=use_stop_list)

    @classmethod
    def from_file(cls, db_path, **kwargs):
        with h5py.File(db_path, 'r') as f:
            db_dict = {key: f[key].value for key in f if not key == 'vocabulary'}
            vocabulary = f['vocabulary'].value

        instance = cls(db_dict, vocabulary, **kwargs)
        return instance

class AnnVisualDatabase(VisualDatabase):
    def __init__(self, image_dict, vocabulary, stop_bottom=0, stop_top=0):
        super().__init__(image_dict, vocabulary, stop_bottom=stop_bottom, stop_top=stop_top)
        self.index = annoy.AnnoyIndex(128, metric='euclidean')
        for i, x in enumerate(vocabulary):
            self.index.add_item(i, x)
        self.index.build(n_trees=20)

    def _descriptor_to_vector(self, des):
        document_word_count = collections.Counter()
        for i, d in enumerate(des):
            l, *_ = self.index.get_nns_by_vector(d, 1)
            document_word_count[l] += 1
        document_word_freq = np.array([document_word_count[i] for i in range(len(self.vocabulary))])
        return document_word_freq



class MultiVisualDatabase(VisualDatabase):
    def __init__(self, databases):
        self.databases = databases

        image_dict = self._build_image_dict()
        self._build_db(image_dict)

        self._gridded = {}

    def _descriptor_to_vector(self, des):
        vectors = [descriptors_to_bow_vector(des, db.vocabulary) for db in self.databases]
        return np.hstack(vectors)

    def _build_image_dict(self):
        # Get list of images
        db0 = self.databases[0]
        keys = db0._tfidf_vectors.keys()

        image_dict = {}
        for key in keys:
            Vdb = np.hstack([db._image_dict[key] for db in self.databases])
            assert Vdb.ndim == 1
            image_dict[key] = Vdb
        return image_dict
def save_keypoints_and_descriptors(path, kps, desc):
    with h5py.File(path, 'w') as f:
        f['descriptors'] = np.vstack(desc)
        g = f.create_group('keypoints')
        g['pt'] = np.vstack([kp.pt for kp in kps])
        g['size'] = np.vstack([kp.size for kp in kps])
        g['angle'] = np.vstack([kp.angle for kp in kps])
        g['response'] = np.vstack([kp.response for kp in kps])
        g['octave'] = np.vstack([kp.octave for kp in kps])