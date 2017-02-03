import collections
import collections.abc
import os

import cv2
import numpy as np
import h5py
import annoy

from .utils import filter_roi, load_descriptors_and_keypoints
from .colornames import calculate_colornames, cname_file_for_image
from vsearch.sift import sift_file_for_image, calculate_sift


def cos_distance(x, y):
    "The cosine angle distance between two vectors of equal size"
    return 1 - np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y))


class DatabaseError(Exception):
    pass

LatLng = collections.namedtuple('LatLng', 'lat lng')
DatabaseEntry = collections.namedtuple('DatabaseEntry', ['key', 'bow', 'latlng'])

SUBCLASS_MESSAGE = "Please use one of the subclasses"


class QueryableDatabase(collections.abc.Mapping):
    """Baseclass for a database that can be queried by an image"""

    def query_image(self, image, roi):
        """Query using an image array and region of interest

        Parameters
        ---------------
        image : np.ndarray
            Image array
        roi : array_like
            Region of interest encoded as [x, y, width, height]

        Returns
        --------------
        Sorted list of database matches [(key1, distance1), (key2, distance2), ...] where distance1 < distance2.
        """
        raise NotImplementedError

    def query_path(self, path, roi):
        """Query using an image path and region of interest

        Parameters
        ---------------
        path : str
            Path to the query image
        roi : array_like
            Region of interest encoded as [x, y, width, height]

        Returns
        --------------
        Sorted list of database matches [(key1, distance1), (key2, distance2), ...] where distance1 < distance2.
        """
        raise NotImplementedError


class BagOfWordsDatabase(collections.abc.MutableMapping):
    """Bag of Words (Bag of Features) database
    """
    def __init__(self, vocabulary):
        """Initialize the database

        Parameters
        -------------
        vocabulary : array_like
            The vocabulary, a KxD array with K words/prototypes of dimensionality D.
        """
        self.image_vectors = {}
        self.idf = None
        self._load_vocabulary(vocabulary)
        self._word_counts = np.zeros(self.vocabulary_size, dtype='int')

    def add_image(self, key, descriptors_or_bow):
        """Add image to the database

        Parameters
        ----------------
        key : str
            The key under which the image will be stored. Usually the filename.
        descriptors_or_bow : array_like
            If 1D it is treated as a single, precomputed BoW-vector.
            If 2D it is treated as a set of feature descriptors, which will be bagged using the database vocabulary.
        """
        if key in self.image_vectors:
            raise DatabaseError("Image '{}' is already in the database".format(key))

        if descriptors_or_bow.ndim == 1:
            bow = descriptors_or_bow
        else:
            descriptors = descriptors_or_bow
            bow = self.bag(descriptors)

        if not len(bow) == self.vocabulary_size:
            raise DatabaseError("Bag of Words vector had wrong size: {:d} (expected {:d})".format(len(bow), self.vocabulary_size))

        self.image_vectors[key] = bow

        # Update IDF
        self._word_counts += (bow > 0)
        self.idf = np.log(len(self.image_vectors) / (1 + self._word_counts).astype('float'))

    @property
    def vocabulary_size(self):
        """Size of the vocabulary"""
        return self._voc_size()

    def _voc_size(self):
        raise NotImplementedError(SUBCLASS_MESSAGE)

    def __len__(self):
        return len(self.image_vectors)

    def __delitem__(self, key):
        del self.image_vectors[key]

    def __getitem__(self, key):
        return self.image_vectors[key]

    def __iter__(self):
        return iter(self.image_vectors)

    def __setitem__(self, key, value):
        if key in self:
            raise DatabaseError("Illegal to change BOW vectors after insertion!")
        else:
            self.add_image(key, value)

    def query_descriptors(self, descriptors):
        """Query the database by a set of descriptors

        Parameters
        ---------------
        descriptors : array_like
            NxD array of N descriptors of dimensionality D (which must match the database vocabulary)

        Returns
        --------------
        Sorted list of database matches [(key1, distance1), (key2, distance2), ...] where distance1 < distance2.
        """
        q_tf = self.bag(descriptors)
        q_tfidf = q_tf * self.idf

        matches = []
        for key, t_tf in self.image_vectors.items():
            t_tfidf = t_tf * self.idf
            distance = cos_distance(q_tfidf, t_tfidf)
            matches.append((key, distance))

        return sorted(matches, key=lambda x: x[1])

    def bag(self, descriptors):
        """Create bag vector from descriptors

        Parameters
        ---------------
        descriptors : array_like
            NxD array of N descriptors of dimensionality D (which must match the database vocabulary)

        Returns
        --------------
        v : array_like
            K-dimensional vector of word frequencies, where K is the size of the vocabulary
        """
        raise NotImplementedError(SUBCLASS_MESSAGE)

    def _load_vocabulary(self, vocabulary):
        raise NotImplementedError(SUBCLASS_MESSAGE)

    @classmethod
    def from_file(cls, database_file):
        """Load database from file"""
        with h5py.File(database_file, 'r') as f:
            vocabulary = f['vocabulary']
            instance = cls(vocabulary)

            for key in f:
                if not key == 'vocabulary':
                    descriptors = f[key].value
                    instance.add_image(key, descriptors)

        return instance


class AnnDatabase(BagOfWordsDatabase):
    """Approximate Nearest Neighbour database

    Instead of computing the exact nearest neighbour, this database will return an approximate answer, but will be much
    faster.

    This implementation usses the annoy NN-library.
    """
    def __init__(self, vocabulary):
        """Initialize the database

        Parameters
        -------------
        vocabulary : array_like
            The vocabulary, a KxD array with K words/prototypes of dimensionality D.
        """
        self.annoy_index = None
        self.n_trees = 20
        super().__init__(vocabulary)

    def _load_vocabulary(self, vocabulary):
        feat_size = vocabulary.shape[1]
        self.annoy_index = annoy.AnnoyIndex(feat_size, metric='euclidean')
        for i, x in enumerate(vocabulary):
            self.annoy_index.add_item(i, x)
        self.annoy_index.build(self.n_trees)

    def _voc_size(self):
        return self.annoy_index.get_n_items()

    def bag(self, descriptors):
        if not descriptors.shape[1] == self.annoy_index.f:
            raise DatabaseError(
                "Descriptor vectors had wrong size: {:d} (expected {:d})".format(descriptors.shape[1], self.annoy_index.f))
        document_word_count = np.zeros(self.vocabulary_size)
        for i, d in enumerate(descriptors):
            l, *_ = self.annoy_index.get_nns_by_vector(d, 1)
            document_word_count[l] += 1
        return document_word_count


class SiftFeatureDatabase(QueryableDatabase, AnnDatabase):
    """An ANN database for SIFT features"""
    def query_image(self, image, roi):
        """Query using an image array and region of interest

        Parameters
        ---------------
        image : np.ndarray
            Image array
        roi : array_like
            Region of interest encoded as [x, y, width, height]

        Returns
        --------------
        Sorted list of database matches [(key1, distance1), (key2, distance2), ...] where distance1 < distance2.
        """
        descriptors, keypoints = calculate_sift(image, roi)
        return self.query_descriptors(descriptors)

    def query_path(self, path, roi):
        sift_file = sift_file_for_image(path)
        if os.path.exists(sift_file):
            print('Loading SIFT features from', sift_file)
            descriptors, keypoints = load_descriptors_and_keypoints(sift_file)
            descriptors, keypoints = filter_roi(descriptors, keypoints, roi)
            return self.query_descriptors(descriptors)
        else:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return self.query_image(image, roi)


class ColornamesFeatureDatabase(QueryableDatabase, AnnDatabase):
    """An ANN database for color names features"""
    def query_image(self, image, roi):
        """Query using an image array and region of interest

            Parameters
            ---------------
            image : np.ndarray
                Image array
            roi : array_like
                Region of interest encoded as [x, y, width, height]

            Returns
            --------------
            Sorted list of database matches [(key1, distance1), (key2, distance2), ...] where distance1 < distance2.
            """
        descriptors, keypoints = calculate_colornames(image, roi)
        return self.query_descriptors(descriptors)

    def query_path(self, path, roi):
        cname_file = cname_file_for_image(path)
        if os.path.exists(cname_file):
            print('Loading Colorname features from', cname_file)
            descriptors, keypoints = load_descriptors_and_keypoints(cname_file)
            descriptors, keypoints = filter_roi(descriptors, keypoints, roi)
            return self.query_descriptors(descriptors)
        else:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return self.query_image(image, roi)


class SiftColornamesWrapper(QueryableDatabase):
    """A database that combines a SIFT and color names databse

    For each query, both the SIFT and color names database will be queried.
    The resulting matches are then sorted using the minimum of the SIFT and color names distance value.
    """
    def __init__(self, sift_db, cname_db):
        self.sift_db = sift_db
        self.cname_db = cname_db
        if not self.sift_db.image_vectors.keys() == self.cname_db.image_vectors.keys():
            raise DatabaseError("SIFT and Colornames databases had different keys!")

    @classmethod
    def from_files(cls, sift_db_path, cname_db_path):
        """Load the database from a SIFT and color names database"""
        sift_db = SiftFeatureDatabase.from_file(sift_db_path)
        cname_db = ColornamesFeatureDatabase.from_file(cname_db_path)
        instance = cls(sift_db, cname_db)
        return instance

    def query_path(self, path, roi):
        sift_matches = dict(self.sift_db.query_path(path, roi))
        cname_matches = dict(self.cname_db.query_path(path, roi))
        return self.combine_matches(sift_matches, cname_matches)

    def query_image(self, image, roi):
        """Query using an image array and region of interest

            Parameters
            ---------------
            image : np.ndarray
                Image array
            roi : array_like
                Region of interest encoded as [x, y, width, height]

            Returns
            --------------
            Sorted list of database matches [(key1, distance1), (key2, distance2), ...] where distance1 < distance2.
            """
        sift_matches = dict(self.sift_db.query_image(image, roi))
        cname_matches = dict(self.cname_db.query_image(image, roi))
        return self.combine_matches(sift_matches, cname_matches)

    def combine_matches(self, sift_matches, cname_matches):
        """Combine SIFT and colornames matches"""
        sift_matches = dict(sift_matches)
        cname_matches = dict(cname_matches)
        matches = [(key, min(sift_matches[key], cname_matches[key])) for key in sift_matches]
        matches.sort(key=lambda x: x[1])
        return matches

    def __getitem__(self, key):
        return self.sift_db[key], self.cname_db[key]

    def __iter__(self):
        return iter(self.sift_db)

    def __len__(self):
        return len(self.sift_db)


class DatabaseWithLocation(QueryableDatabase):
    """A database that also contains location data"""
    def __init__(self, visualdb):
        super().__init__()
        self.visualdb = visualdb
        self.locations = collections.defaultdict(lambda: None)

    def __delitem__(self, key):
        del self.visualdb[key]
        del self.locations[key]

    def __getitem__(self, key):
        e = DatabaseEntry(key, self.visualdb[key], self.locations[key])
        return e

    def __iter__(self):
        return iter(self.visualdb)

    def __len__(self):
        return len(self.visualdb) if self.visualdb is not None else 0

    def __setitem__(self, key, latlng):
        self.locations[key] = latlng

    def query_image(self, image, roi):
        """Query using an image array and region of interest

            Parameters
            ---------------
            image : np.ndarray
                Image array
            roi : array_like
                Region of interest encoded as [x, y, width, height]

            Returns
            --------------
            Sorted list of database matches [(key1, distance1), (key2, distance2), ...] where distance1 < distance2.
            """
        visual_matches = self.visualdb.query_image(image, roi)
        matches = [(self[key], score) for key, score in visual_matches]
        return matches

    def query_path(self, path, roi):
        visual_matches = self.visualdb.query_path(path, roi)
        matches = [(self[key], score) for key, score in visual_matches]
        return matches