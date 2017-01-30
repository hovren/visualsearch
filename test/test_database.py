import unittest

import numpy as np
import numpy.testing as nt
import h5py

from vsearch.database import AnnDatabase, DatabaseError, DatabaseWithLocation, DatabaseEntry, LatLng

test_db = 'test_db.h5'
test_db_items = 222

test_vocabulary = 'test_voc.h5'
test_vocabulary_size = 100
test_vocabulary_feature_size = 11

class AnnDatabaseTest(unittest.TestCase):
    def setUp(self):
        with h5py.File(test_vocabulary, 'r') as f:
            self.vocabulary = f['vocabulary'].value

    def random_bow(self):
        bow = np.random.randint(-4, 5, size=test_vocabulary_size)
        bow[bow < 0] = 0
        return bow

    def descriptors_from_bow(self, bow):
        return np.vstack([np.tile(self.vocabulary[i], (n, 1)) for i, n in enumerate(bow)])

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
        expected_bow = self.random_bow()
        descriptors = self.descriptors_from_bow(expected_bow)

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

    def test_vocabulary_size(self):
        db = AnnDatabase(self.vocabulary)
        self.assertEqual(db.vocabulary_size, test_vocabulary_size)

    def test_idf_changes(self):
        db = AnnDatabase(self.vocabulary)
        self.assertIsNone(db.idf)

        bows = np.vstack([self.random_bow() for _ in range(7)])

        for i, bow in enumerate(bows):
            key = 'test_image_{:d}.jpg'.format(i)
            descriptors = self.descriptors_from_bow(bow)
            db.add_image(key, descriptors)

            N = i + 1
            expected_idf = np.log(N / (1 + np.sum(bows[:N].astype('bool'), axis=0).astype('float')))
            nt.assert_almost_equal(db.idf, expected_idf)

    def test_add_bad_bow_size(self):
        db = AnnDatabase(self.vocabulary)
        bad_bow_size = 13
        self.assertNotEqual(bad_bow_size, db.vocabulary_size)
        bad_bow = np.random.uniform(-5, 5, size=bad_bow_size)
        with self.assertRaises(DatabaseError):
            db.add_image('somelabel', bad_bow)

    def test_add_bad_descriptor_size(self):
        db = AnnDatabase(self.vocabulary)
        bad_descriptor_size = 13
        num_descriptors = 23
        bad_descriptors = np.random.uniform(-5, 5, size=(num_descriptors, bad_descriptor_size))

        with self.assertRaises(DatabaseError):
            db.add_image('somelabel', bad_descriptors)

    def test_length(self):
        db = AnnDatabase(self.vocabulary)
        N = 6
        for i in range(N):
            key = 'test_image_{}.jpg'.format(i)
            des = self.descriptors_from_bow(self.random_bow())
            db.add_image(key, des)
        self.assertEqual(len(db), N)

    def test_load_from_file(self):
        db = AnnDatabase.from_file(test_db)
        self.assertEqual(len(db), test_db_items)

class LocationDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.visualdb = AnnDatabase.from_file(test_db)

    def test_no_locations(self):
        locdb = DatabaseWithLocation(self.visualdb)
        for val in locdb.values():
            self.assertIsInstance(val, DatabaseEntry)
            self.assertIsNone(val.latlng)

    def test_random_locs(self):
        locdb = DatabaseWithLocation(self.visualdb)
        lat, lng = 58.0, 65.4
        for key, val in locdb.items():
            locdb[key] = LatLng(lat, lng)

        for key, val in locdb.items():
            self.assertEqual(val.latlng.lat, lat)
            self.assertEqual(val.latlng.lng, lng)

