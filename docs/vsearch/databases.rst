Bag of Words databases
=========================
The visual search is performed using a technique called *Bag of Words*
or *Bag of Features*.

Examples
-------------------------

Basic searching
^^^^^^^^^^^^^^^

The default visual database class is :class:`.SiftColornamesWrapper`, which is a convenience wrapper for making combined
searches in a SIFT and color names database.

::

    from vsearch.database import SiftColornamesWrapper

    # Load the database
    database = SiftColornamesWrapper.from_files('my_sift_db.h5',
                                                'my_cname_db.h5')

    # Define a region of interest
    roi = [x, y, width, height]

    # Query by path to image
    matches = database.query_path('some_image.jpg', roi, max_result=5)

    # Query by image
    image = ...
    matches = database.query_image(image, roi, max_distance=0.8)

Matches are returned as a list of `(key, distance)` tuples, sorted by distance (ascending).

Dealing with location data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adding location data to the database is done using the :class:`.DatabaseWithLocation` wrapper class.
It uses a :class:`.DatabaseEntry`   class which contains location in the form of :class:`.LatLng` objects for location data.
Accessing entries is done like any dictionary/mapping::

    from vsearch.database import SiftColornamesWrapper, DatabaseWithLocation

    visualdatabase = SiftColornamesWrapper.from_files('my_sift_db.h5',
                                                'my_cname_db.h5')

    locdatabase = DatabaseWithLocation(visualdatabase)
    entry = locdatabase['some_key']
    print('{} location is lat={:.6f}, lng={:.6f}'.format(entry.key, entry.latlng.lat, entry.latlng.lng))

We can also update the location::

    new_location = LatLng(12.3456, 45.678)
    locdatabase['some_key'] = new_location

Visual databases
-------------------------
.. autoclass:: vsearch.database.SiftColornamesWrapper
    :members:

.. autoclass:: vsearch.database.SiftFeatureDatabase
    :members:

.. autoclass:: vsearch.database.ColornamesFeatureDatabase
    :members:

Location database
-------------------
.. autoclass:: vsearch.database.DatabaseWithLocation
    :members:

Types
-------------------
.. autoclass:: vsearch.database.DatabaseError

.. autoclass:: vsearch.database.LatLng

.. autoclass:: vsearch.database.DatabaseEntry
