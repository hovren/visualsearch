The Visual Search tool GUI
=====================================
The visual search tool GUI is what you would use to query the database.

Start the application
----------------------------

To start the application simply type ``vsearch_query`` on the command line.

If you want, you can specify the SIFT and color names databases to use from the command line, together with the
path to the image directory that contains the database images::

    vsearch_query my_sift_database.h5 my_cname_database.h5 /path/to/images

Overview
------------------------
The left part of the application is dedicated to a map of the area.
Large blue markers on the map represents geo-tagged images in the database.
The markers are clickable to select the corresponding image.

The right sidebar can change between the *Database view* and the *Query view*.

Database View
^^^^^^^^^^^^^
In the database view you can load the database(s), and also update image locations.

The list view shows loaded database images. If the image has a geographic location set, the list entry shows a small
"globe" icon.

Loading a database
"""""""""""""""""""""""
If you did not load the databases from the command line you need to load one by pressing the *Load Database* button.
This allows you to choose the following:

- The SIFT database
- The color names database
- The directory that contains the images indexed by the above databases

If there is a file called ``geo.csv`` in the image root directory, the image locations stored in it will be read.

After loading the database, the keys (filenames) are shown in the list, and a marker is created on the map for each image.
Selecting an image by clicking on the map, or in the list view, shows the image in the lower part of the sidebar.

Updating location
"""""""""""""""""""""""""
After an image has been selected, its marker on the map can be dragged to a new position.
The new location is used for the rest of the session, but is **not saved to disk** until the *Save Locations* button is pressed.

Query view
^^^^^^^^^^^^^^^^^^^^^^^^^^
The query view is where query results are shown. If no query has been made yet, a new query can also be started from here.

The top part shows the current image patch (region of interest) that was used for the query.
In the middle is a list of results which shows the database key and its corresponding *similarity* measurement.
The bottom part shows the result image that is currently selected.

Making a query
-------------------------
There are two ways to start a new query

1. Pressing the *Query* button from the Query view. This can be used to query an image that is not in the database yet.
2. In the Database view, select an image in the database and press the *Use as Query* button.

In both cases the Query dialog is opened.

To make the query from the query dialog do the following

1. Select an image from the filesystem using the *Select file* button. (If you used the *Use as Query* button, this is done for you already.)
2. In the image, drag a rectangle to define the region of interest.
3. Select result filtering criteria. You can choose to return either a * maximum number of results*, or images that pass a *similarity* threshold, or both.

Similarity
^^^^^^^^^^^^^^^
The similarity score is simply ``1 - distance`` where ``distance`` is cosine angle distance between BoF-vectors.

