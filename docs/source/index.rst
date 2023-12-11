.. KerrGeoPy documentation master file, created by
   sphinx-quickstart on Wed Jun 21 09:39:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========

**KerrGeoPy** is a python library for computing bound timelike geodesics in Kerr spacetime. It is intended for use in computing orbital trajectories for extreme-mass-ratio inspirals (EMRIs). 
It implements the analytical solutions for plunging orbits from `Dyson and van de Meent <https://arxiv.org/abs/2302.03704>`_, as well as solutions for stable orbits from `Fujita and Hikida <https://arxiv.org/abs/0906.1420>`_. 
The library also provides a set of methods for computing constants of motion and orbital frequencies, and can generate plots and animations like those shown below.

.. image:: images/thumbnail.png
    :align: left
    :width: 45%

.. image:: images/thumbnail.gif
    :align: right
    :width: 45%

.. _Installation:

Installation
------------
Install using Anaconda

.. code-block:: bash

   conda install -c conda-forge kerrgeopy

or using pip

.. code-block:: bash

   pip install kerrgeopy

.. note::
   KerrGeoPy uses functions introduced in scipy 1.8, so it may also be necessary to update scipy by running :code:`pip install scipy -U`, although in most cases this should be done automatically by pip.
   Certain plotting and animation functions also make use of features introduced in matplotlib 3.7 and rely on `ffmpeg <https://ffmpeg.org/download.html>`_, which can be easily installed using `homebrew <https://formulae.brew.sh/formula/ffmpeg>`_ or `anaconda <https://anaconda.org/conda-forge/ffmpeg>`_.

See `Getting Started <notebooks/Getting%20Started.html>`_ for basic usage. See the `API Reference`_ below or the `Modules <_autosummary/kerrgeopy.html>`_ page for a complete list of classes and methods.


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/Getting Started
   notebooks/Orbital Properties
   notebooks/Trajectory
   notebooks/Graphics
   
.. _API Reference:

API Reference
-------------

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   Modules <https://kerrgeopy.readthedocs.io/en/latest/_autosummary/kerrgeopy.html>

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   
   ~kerrgeopy.stable.StableOrbit
   ~kerrgeopy.plunge.PlungingOrbit
   ~kerrgeopy.orbit.Orbit
   ~kerrgeopy.spacetime.KerrSpacetime
   ~kerrgeopy.constants
   ~kerrgeopy.frequencies
   ~kerrgeopy.initial_conditions
   ~kerrgeopy.units

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
