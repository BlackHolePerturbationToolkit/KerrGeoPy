from setuptools import setup
import setuptools

setup(
    name="kerrgeopy",
    version="0.0.1",
    author="Seyong Park",
    description="Library for computing bound and plunging geodesics in Kerr spacetime",
    url="https://github.com/syp2001/KerrGeoPy",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Natural Language :: English",
    ),
    install_requires=["scipy","numpy","matplotlib"]
)