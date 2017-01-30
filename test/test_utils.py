import unittest
import tempfile
import os

from vsearch.utils import image_for_descriptor_file

class UtilTests(unittest.TestCase):
    def test_image_for_descriptor(self):
        with tempfile.TemporaryDirectory() as tempdir:
            for fname in ['image_1.jpg', 'image_1.sift.h5',
                          'image_2.sift.h5',
                          'image_3.jpg', 'image_3.png', 'image_3.sift.h5']:
                open(os.path.join(tempdir, fname), 'a').close()

            image_file = image_for_descriptor_file(os.path.join(tempdir, 'image_1.sift.h5'))
            self.assertEqual(image_file, os.path.join(tempdir, 'image_1.jpg'))

            # Missing image
            with self.assertRaises(ValueError):
                image_file = image_for_descriptor_file(os.path.join(tempdir, 'image_2.sift.h5'))

            # Multiple candidates
            with self.assertRaises(ValueError):
                image_file = image_for_descriptor_file(os.path.join(tempdir, 'image_3.sift.h5'))
