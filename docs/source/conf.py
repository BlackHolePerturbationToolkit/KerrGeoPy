# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'KerrGeoPy'
copyright = '2023, Seyong Park'
author = 'Seyong Park'

# The full version, including alpha/beta/rc tags
release = '0.9.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_nb",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]
napoleon_use_ivar = True
myst_enable_extensions = ["dollarmath", "amsmath"]
nb_execution_mode = "off"
autosummary_generate = True
# https://stackoverflow.com/questions/2701998/automatically-document-all-modules-recursively-with-sphinx-autodoc/62613202#62613202

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

html_theme_options = {
    "repository_url": "https://github.com/BlackHolePerturbationToolkit/KerrGeoPy",
    "use_repository_button": True,
    "navbar_end": ["navbar-icon-links"],
}
html_title = "KerrGeoPy"
html_context = {
    "default_mode": "light"
}