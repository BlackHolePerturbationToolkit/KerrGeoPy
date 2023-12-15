---
title: '`KerrGeoPy`: A Python Package for Computing Timelike Geodesics in Kerr Spacetime'
tags:
    - Python
    - black holes 
    - perturbation theory
    - gravitational waves
authors:
    -   name: Seyong Park
        orcid: 0009-0002-1152-9324
        affiliation: "1, 2"
    -   name: Zachary Nasipak
        orcid: 0000-0002-5109-9704
        affiliation: 1
affiliations:
    -   name: NASA Goddard Space Flight Center, Greenbelt, MD, USA
        index: 1
    -   name: University of Maryland, College Park, MD, USA
        index: 2
date: 15 December 2023
bibliography: paper.bib
---

# Summary

In general relativity, the motion of a free falling test particle in a curved spacetime is 
described by a geodesic - the generalization of a "straight line" path to a curved space. 
The geodesics of Kerr spacetime are of particular interest in the field of black 
hole perturbation theory because they describe the zeroth order motion of a small object 
moving through the background spacetime of a large spinning black hole. For this reason, computing
geodesics is an important step in modelling the gravitational radiation emitted by an
extreme mass ratio inspiral (EMRI) - an astrophysical binary in which a stellar mass
compact object, such as a neutron star or a small ($10^1 - 10^2 M_\odot$) black hole, 
spirals into a massive ($10^4 - 10^7 M_\odot$) black hole.

Kerr spacetime has several nice properties which simplify the problem of computing geodesics. Since 
it has both time translation symmetry and rotational symmetry, energy and angular momentum are conserved quantities. It is also
equipped with a higher order symmetry which gives rise to a third constant of motion called the Carter
constant. These three constants of motion, along with the spin of the black hole, uniquely define a geodesic up to 
initial conditions [@schmidt].Alternatively, geodesics can be identified using a suitably generalized 
version of the parameters used to define a Keplerian orbit (eccentricity, semi-latus rectum and inclination angle). 
Bound geodesics also possess fundamental frequencies since their radial, azimuthal and polar motion is periodic.

`KerrGeoPy` is a Python implementation of the `KerrGeodesics` [@kerrgeodesics] Mathematica library 
which computes both stable and plunging geodesics in Kerr spacetime using the 
analytic solutions to the geodesic equation derived in [@fujita] and 
[@dyson]. These solutions are written in terms of elliptic integrals, which are 
evaluated using `SciPy`. Users can construct a geodesic by providing the initial position and
four-velocity, or by providing either the constants of motion or the Keplerian parameters described above. 
`KerrGeoPy` also provides methods for computing the fundamental frequencies 
and constants of motion associated with a given geodesic and implements the algorithm described 
in [@stein] for finding the location of the last stable orbit orbit, known as the separatrix.

![Example of an equatorial (left), spherical (center) and generic (right) orbit computed by `KerrGeoPy`](orbits.png)


# Statement of Need

EMRIs are expected to be a major source observable by the Laser Inteferometer Space 
Antenna (LISA), a future space-based gravitational wave observatory consisting of a triangular 
constellation of three satellites in orbit around the sun. LISA is an ESA-led mission 
with significant contributions from NASA which is set to launch in the 2030s. It will
complement existing ground-based detectors by opening up the millihertz band of the 
gravitational wave spectrum [@lisa]. Because sources in this band evolve slowly over time and remain observable 
for a period of days to years, LISA is expected to detect many overlapping signals at all times. 
Thus, accurate waveform models are needed in order to identify gravitational wave sources and 
perform parameter estimation - the process of approximating characteristics of a source.

For most LISA sources, well-developed waveform models based on either numerical relativity 
or post-Newtonian theory already exist. However, EMRIs are instead best 
described by black hole perturbation theory, and the EMRI waveform models that currently exist 
are comparatively underdeveloped. In a perturbation theory model, the orbital trajectory is assumed to be a geodesic at 
leading order. Higher order corrections are then computed by introducing the gravitational 
field of the inspiralling object as a perturbation to the background spacetime of the massive black hole, 
expanded in powers of the mass ratio.

In order to meet the accuracy requirements for LISA parameter estimation, EMRI waveform 
models must include second order corrections to the orbital trajectory. However, to date, 
most development has focused on first order corrections to only the simplest cases, 
such as Schwarzschild orbits and equatorial Kerr orbits [@emri]. Additionally, most development 
has taken place in Mathematica, which is an expensive and proprietary piece of software. `KerrGeoPy` is 
intended to support future development of higher-order waveform models in preparation for
LISA by providing a free alternative to the existing `KerrGeodesics` Mathematica library for other
researchers to build on in their own projects.

Although other Python packages [@kerrgeodesicgw] with similar functionality do exist, they mostly rely on numerical 
integration to compute geodesics. The analytic solutions used by `KerrGeoPy` have two main advantages
over this approach. First, they are guaranteed to be numerically stable and can be quickly evaluated at
any point in time. Second, they produce several useful intermediate terms which cannot easily be
computed using numerical integration.

`KerrGeoPy` is a part of the [Black Hole Perturbation Toolkit](https://bhptoolkit.org). The source code
is hosted on [Github](https://github.com/BlackHolePerturbationToolkit/KerrGeoPy) and the package is
distributed through both [PyPI](https://pypi.org/project/kerrgeopy/) and [conda-forge](https://anaconda.org/conda-forge/kerrgeopy).
Automated unit tests are run using [Github Actions](https://github.com/BlackHolePerturbationToolkit/KerrGeoPy/actions/workflows/tests.yml) and
comprehensive documentation is available on [ReadTheDocs](https://kerrgeopy.readthedocs.io/).

# Software Citations

`KerrGeoPy` has the following dependencies:

- `NumPy` [@numpy]
- `SciPy` [@scipy]
- `Matplotlib` [@matplotlib]
- `tqdm` [@tqdm]

# Acknowledgements

We would like to thank Niels Warburton and Barry Wardell for their assistance in releasing 
`KerrGeoPy` as part of the Black Hole Perturbation Toolkit.

# References