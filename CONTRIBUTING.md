# Contributing to KerrGeoPy

Contributions are welcome and greatly appreciated! There are many different ways to contribute including writing tutorials, improving the documentation, submitting bug reports and suggesting or implementing new features.

## Types of Contributions

### Bug Reports and Feature Requests

If you identify a bug or have an idea for a new feature, open an issue at https://github.com/BlackHolePerturbationToolkit/KerrGeoPy/issues. 
For bug reports, please also include your operating system/processor architecture and any relevant information about your python environment.

### Documentation

Improvements to the documentation are always welcome. Documentation for KerrGeoPy is generated from python docstrings using [Sphinx](https://www.sphinx-doc.org/en/master/index.html) and hosted on [ReadTheDocs](https://docs.readthedocs.io/en/stable/). Docstrings should follow the [numpydoc](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) format. The tutorial pages are generated from the Jupyter notebooks in https://github.com/BlackHolePerturbationToolkit/KerrGeoPy/tree/main/docs/source/notebooks using [MyST-NB](https://myst-nb.readthedocs.io/en/latest/).


### Contributing Code

To contribute code, follow the steps below to submit a pull request. See https://bhptoolkit.org/conventions.html for a list of naming conventions used throughout the package.
Ideally, any new features should also come with unit tests and documentation. See https://github.com/BlackHolePerturbationToolkit/KerrGeoPy/tree/main/tests for more information about testing.

## Getting Started

First, fork/clone the repository and create a new branch. After implementing changes, verify that all unit tests pass.

```bash
python -m unittest
```

To preview any changes to the documentation, build the docs

```bash
cd docs
make html
```

Finally, push the changes and submit a pull request through Github.

## Community

KerrGeoPy is a part of the [Black Hole Perturbation Toolkit](https://bhptoolkit.org/index.html). See the [Users and Contributors](https://bhptoolkit.org/users.html) page for more information.