Create a searchable database
================================

To create a Bag of Words database the following needs to be done

1. Compute visual features (SIFT and color names)
2. Compute a visual vocabulary for the features
3. Create the Bag of Words database using the vocabularies

Visual features
--------------------------------
The visual features used are SIFT and color names.
For this, two scripts are provided: ``vsearch_sift`` and ``vsearch_colornames``.

SIFT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To compute SIFT features run::

    vsearch_sift /path/to/images

This will create a number of ``XXXX.sift.h5`` files where ``XXXX`` is the filename of an image, without the extension.
The descriptor file contains both SIFT descriptors and keypoints.

Color names
^^^^^^^^^^^^^^^^^^^^^
To compute color names features run::

    vsearch_colornames /path/to/images

This will create a number of ``XXXX.cname.h5`` files where ``XXXX`` is the filename of an image, without the extension.
If there is already an ``XXXX.sift.h5`` file in the directory, those SIFT keypoints are used, otherwise new SIFT keypoints
are computed for this image (but not descriptors, and the ``XXXX.sift`` file is not created!)

Visual vocabulary
----------------------------------
To create a visual vocabulary using SIFT features, with 50000 words run::

    vsearch_vocabulary /path/to/images /path/to/output 50000 sift --iterations=15 --tries=100

To instead use color name features, replace ``sift`` with ``colornames``.
This tool uses the *K-means* algorithm to cluster the data.
The ``--iterations`` flag tells how many K-means iterations to run. 15 should suffice in most cases.
The ``--tries`` flag controls the number of times the K-means algorithm is started, using a new randomly selected set of
seed points.
Larger vocabularies need a larger value.

Note that running K-means on a large dataset can take a lot of time!

The Database
----------------------------------------
Creating a visual database requires a directory of images that should be put into the database, and a vocabulary.

To create a database from SIFT features run::

    vsearch_database /path/to/images /path/to/vocabulary /path/to/output sift

For a color names database, replace ``sift`` with ``colornames``.
