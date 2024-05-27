from setuptools import setup
import setuptools
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="kerrgeopy",
    version="0.9.2",
    author="Seyong Park",
    description="Library for computing stable and plunging geodesics in Kerr spacetime",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/BlackHolePerturbationToolkit/KerrGeoPy",
    packages=setuptools.find_packages(exclude=["tests"]),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Natural Language :: English",
    ),
    install_requires=["scipy>=1.8","numpy","matplotlib>=3.7","tqdm"]
)