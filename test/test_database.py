import unittest

import numpy as np
import numpy.testing as nt
import h5py

from vsearch.database import AnnDatabase, DatabaseError

test_vocabulary = 'test_voc.h5'
test_vocabulary_size = 100
test_vocabulary_feature_size = 11

class AnnDatabaseTest(unittest.TestCase):
    def setUp(self):
        with h5py.File(test_vocabulary, 'r') as f:
            self.vocabulary = f['vocabulary'].value

    def test_empty_database(self):
        db = AnnDatabase(self.vocabulary)
        self.assertIsNotNone(db.annoy_index)
        self.assertEqual(db.annoy_index.f, test_vocabulary_feature_size)
        self.assertEqual(db.annoy_index.get_n_items(), test_vocabulary_size)
        self.assertEqual(len(db.image_vectors), 0)
        self.assertIsNone(db.idf)

    def test_add_bag(self):
        db = AnnDatabase(self.vocabulary)
        test_key = 'some_image.jpg'

        expected_bow = np.random.randint(-4, 5, size=test_vocabulary_size)
        expected_bow[expected_bow < 0] = 0

        descriptors = np.vstack([np.tile(self.vocabulary[i], (n, 1)) for i, n in enumerate(expected_bow)])
        self.assertEqual(descriptors.shape, (np.sum(expected_bow), test_vocabulary_feature_size))
        db.add_image(test_key, descriptors)
        self.assertEqual(len(db.image_vectors), 1)
        nt.assert_equal(db.image_vectors[test_key], expected_bow)

    def test_add_twice(self):
        db = AnnDatabase(self.vocabulary)
        test_key = 'some_image.jpg'
        des = np.random.uniform(-5, 5, size=(20, test_vocabulary_feature_size))

        db.add_image(test_key, des)
        with self.assertRaises(DatabaseError):
            db.add_image(test_key, des)








