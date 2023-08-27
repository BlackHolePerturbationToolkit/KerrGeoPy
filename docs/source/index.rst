.. KerrGeoPy documentation master file, created by
   sphinx-quickstart on Wed Jun 21 09:39:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========

**Kerrgeopy** is a python library for computing geodesics in Kerr spacetime. It is intended for use in computing orbital trajectories for extreme-mass-ratio inspirals (EMRIs). 
It implements the analytical solutions for plunging orbits from `Dyson and van de Meent <https://arxiv.org/abs/2302.03704>`_, as well as solutions for non-plunging orbits from `Fujita and Hikida <https://arxiv.org/abs/0906.1420>`_. 
The library also provides a set of methods for computing constants of motion and orbital frequencies, and can generate plots and animations like those shown below.

.. image:: images/orbit.png
    :align: left
    :width: 45%

.. image:: images/orbit.gif
   :align: right
   :width: 45%

Installation
------------
Install the latest version of kerrgeopy using pip

.. code-block:: bash

   pip install kerrgeopy

See `Getting Started <notebooks/Tutorial.html>`_ for usage examples.

.. note::

   This project is under active development. The documentation is not yet complete.


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/Tutorial

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :caption: API Reference
   :recursive:

   kerrgeopy


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
