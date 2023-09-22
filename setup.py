from setuptools import setup
import setuptools

setup(
    name="kerrgeopy",
    version="0.1.0",
    author="Seyong Park",
    description="Library for computing bound and plunging geodesics in Kerr spacetime",
    url="https://github.com/syp2001/KerrGeoPy",
    packages=setuptools.find_packages(exclude=["tests"]),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Natural Language :: English",
    ),
    install_requires=["scipy>=1.8","numpy","matplotlib>=3.3"]
)