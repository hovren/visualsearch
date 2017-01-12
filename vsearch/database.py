import collections

import numpy as np
import h5py
import annoy


def cos_distance(x, y):
    return 1 - np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y))


class DatabaseError(Exception):
    pass


SUBCLASS_MESSAGE = "Please use one of the subclasses"

class BaseDatabase:
    def __init__(self, vocabulary):
        self.image_vectors = {}
        self.idf = None
        self._load_vocabulary(vocabulary)
        self._word_counts = np.zeros(self.vocabulary_size, dtype='int')

    def add_image(self, key, descriptors_or_bow):
        if key in self.image_vectors:
            raise DatabaseError("Image '{}' is already in the database".format(key))

        if descriptors_or_bow.ndim == 1:
            bow = descriptors_or_bow
        else:
            bow = self.bag(descriptors_or_bow)

        self.image_vectors[key] = bow

        # Update IDF
        self._word_counts += (bow > 0)
        self.idf = np.log(len(self.image_vectors) / (1 + self._word_counts).astype('float'))

    @property
    def vocabulary_size(self):
        return self._voc_size()

    def _voc_size(self):
        raise NotImplementedError(SUBCLASS_MESSAGE)

    def __len__(self):
        return len(self.image_vectors)

    def query(self, descriptors):
        q_tf = self.bag(descriptors)
        q_tfidf = q_tf * self.idf

        matches = []
        for key, t_tf in self.image_vectors.items():
            t_tfidf = t_tf * self.idf
            distance = cos_distance(q_tfidf, t_tfidf)
            matches.append((key, distance))

        return sorted(matches, key=lambda x: x[1])

    def bag(self, descriptors):
        raise NotImplementedError(SUBCLASS_MESSAGE)

    def _load_vocabulary(self, vocabulary):
        raise NotImplementedError(SUBCLASS_MESSAGE)

    @classmethod
    def from_file(cls, database_file):
        with h5py.File(database_file, 'r') as f:
            vocabulary = f['vocabulary']
            instance = cls(vocabulary)

            for key in f:
                if not key == 'vocabulary':
                    descriptors = f[key].value
                    instance.add_image(key, descriptors)

        return instance


class AnnDatabase(BaseDatabase):
    def __init__(self, vocabulary):
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
        document_word_count = np.zeros(self.vocabulary_size)
        for i, d in enumerate(descriptors):
            l, *_ = self.annoy_index.get_nns_by_vector(d, 1)
            document_word_count[l] += 1
        return document_word_count
