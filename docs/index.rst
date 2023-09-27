.. scPCA documentation master file, created by
   sphinx-quickstart on Sat Sep  9 15:05:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to scPCA's documentation!
=================================

scPCA is a versatile matrix factorisation framework designed to analyze single-cell data across diverse experimental designs.

.. image:: https://github.com/sagar87/scPCA/blob/main/docs/scpca_schematic.png?raw=true
    :alt: scPCA schematic

*scPCA is a young project and breaking changes are to be expected.*


Quick install
-------------

scPCA makes use `torch`, `pyro` and `anndata`. We highly recommend to run scPCA on a GPU device.

Via Pypi
^^^^^^^^

The easiest option to install `scpca` is throug Pypi. Simply type

.. code-block:: console
   $ pip install scpca



into your shell and hit enter.

* Free software: MIT license
* Documentation: https://sagar87.github.io/scPCA/index.html

Credits
-------

* Harald Vöhringer


.. toctree::
   :maxdepth: 1
   :caption: API

   models
   tools
   plots

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   notebooks/kang


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
