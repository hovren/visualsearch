.. Visual Search tool documentation master file, created by
   sphinx-quickstart on Thu Feb  2 16:58:37 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Visual Search tool's documentation!
==============================================

The visual search tool is meant to help finding some object, like a person,
in a database of images.
It achieves this by using a technique known as *Bag of Words*.

This tool was developed within the `Visual Sweden <http://visualsweden.se>`_ initiative.

The source repository for this project is hosted on `Github <https://github.com/hovren/visualsearch>`_.

Installation
------------------
The dependencies of this package are

- python 3.4+
- annoy
- opencv
- scipy
- h5py
- PyQt5
- tqdm (only for database creation)
- sklearn (only for database creation)

After meeting the above requirements the `vsearch` python library, and the scripts can be installed
by running::

     python3 setup.py install

Both library and GUI should work on all platforms on which its dependencies are satisfied,
but has until now only been tested on 64-bit Linux (Fedora 25) using the Anaconda Python distribution.

Usage
------------------
.. toctree::
   :maxdepth: 2

   usage/query_gui
   usage/create

Programming API
-------------------
.. toctree::
   :glob:
   :maxdepth: 2

   vsearch/*


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

