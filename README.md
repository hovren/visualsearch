# Visual Search tool
This tool can be used to perform *visual search*, which means that given a region of interest in
an input image, it will find other images in a database that has a high probability to contain that object.
Images stamped with their geographic location are shown on a map of the area.

This work was done as a project within [Visual Sweden](http://www.visualsweden.se/) initiative
in December 2016 - January 2017.

This package is made up from three parts:
- The GUI tool
- Scripts for creating BoF-vocabularies and databases
- A library for querying BoF databases

## Installation
The dependencies of this package are
- python 3.4+
- annoy
- opencv
- scipy
- h5py
- tqdm
- sklearn (only for database creation)

After meeting the above requirements the `vsearch` python library, and the scripts can be installed
by running

    $ python3 setup.py install
    
## Running the GUI tool
The GUI tool is launched by calling

    $ vsearch_query

If no arguments are given it will load empty and you can then select the database to load from
within the application.

Databases can also be selected from the command line by running

    $ vsearch_query /path/to/sift_databae /path/to/colornames_database /path/to/images
 
## Creating a Bag of Features database
To create your own database from a directory with images perform the following steps

1. Compute SIFT features (and keypoints)
```
$ vsearch_sift /path/to/images
```

2. Compute color name features (this uses the SIFT keypoints from the previous step)
```
$ vsearch_colornames /path/to/images
```

3. Compute a SIFT vocabulary. Larger vocabularies are in general better.
A size of 50k seems to work OK on our test dataset:
```
$ vsearch_vocabulary /path/to/images /path/to/output 50000 sift --iterations=15 --tries=100
```
4. Compute a color names vocabulary. We used a size of 15k:
```
$ vsearch_vocabulary /path/to/images /path/to/output 15000 colornames --iterations=15 --tries=100
```
 
5. Compute a SIFT image database
```
$ vsearch_database /path/to/images /path/to/vocabulary /path/to/output sift
```

6. Compute a color names image database
```
$ vsearch_database /path/to/images /path/to/vocabulary /path/to/output colornames
```

7. (Optional) Remove the SIFT and color names vocabularies since they are stored explicitly in
the database files anyway.

## How does it work?
To perform visual search this tool uses the well-known *Bag of Words* or *Bag of Features* method.
Given a *vocabulary* of prototypes in some feature space each 
image is described by how many times these *visual words* appear, in the form of
a word frequency vector.

We can then compute the distance between this vector and all others in the database to
find the most similar ones.
To increase robustness the vectors are weighted using the *term frequency* and
*inverse document frequency* (TF-IDF).
The distance measure we use is cosine-angle between the two vectors (i.e. dot product between
the two vectors, after normalization to unit norm).

We use two databases, one using SIFT, and the other color names features.
During runtime, both databases are queried and the matches are sorted using the
smaller of the two distances.

## Known issues
- Requires Internet connection to download map tiles and load the map javascript libraries.
- Currently uses map tiles from the Open Streetmap Project, which is not advisable for heavy use.
Please see the Open Streetmap Tile Policy for guidance.
- The GUI app is hardcoded to launch the map view at Norrköping, Sweden.
If you use a database from somewhere else please update the starting location.
An even better solution would be to look at the database entries and dynamically
change location.
- All database entries must have a location set, or querying is likely to misbehave
or simply crash.

## License
All code is licensed under the GPLv3, copyright Hannes Ovrén.
 
## Acknowledgements
- This work was funded by [Visual Sweden](http://www.visualsweden.se/)
- The color names lookup table was taken from the homepage of 
[Joost van de Weijer](http://lear.inrialpes.fr/people/vandeweijer/)