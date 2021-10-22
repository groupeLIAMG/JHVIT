<p align="center"> <img src="https://github.com/groupeLIAMG/JHVIT/raw/main/images/JHVIT_Logo.gif" width="400" height="400">
</p>

##

[![pypi](https://img.shields.io/pypi/v/JHVIT.svg)](https://pypi.org/project/JHVIT/)
[![License: MIT](https://img.shields.io/badge/License-MIT%20-blue.svg)](./LICENSE.txt)
[![Documentation Status](https://readthedocs.org/projects/jhvit/badge/?version=latest)](https://jhvit.readthedocs.io/en/latest/)


JHVIT : Joint Hypocenter Velocity Inversion on Tetrahedral meshes


- [The JHVIT package](#heading)
- [Installation and Requirements](#heading)
- [Callable functions](#heading)
- [Examples](#heading)
- [References](#heading)

<!-- toc -->

## The JHVIT package

JHVIT is a python package for locating seismic event hypocenters on unstructured
grids. The package is an extension of the joint hypocenter-velocity inversion method
on tetrahedral meshes. It is mainly recommended for domains with steep topography,
underground cavities and stratigraphic and abnormal geological contacts such as
folds, faults and shear zones. The code is able to locate a wide range of seismic
events going from major earthquakes and nuclear explosions to low and negative
magnitude events. Target application areas include computational seismology,
hydraulic fracture and microseismic monitoring of mining environments or in civil
engineering projects.

The package is written in Python, uses an optimized c++ raytracing code wrapped in
Cython and supports parallel computing.

Documentation is available here: https://jhvit.readthedocs.io/en/latest/ 

## Installation and Requirements

Launch a command line on your device and run:
pip install JHVIT

Requirements:
- The package ttcrpy must be installed in order to perform the raytracing step.
This package can be installed from: https://pypi.org/project/ttcrpy/
- Install a proper version of vtk : https://pypi.org/project/vtk/
- To prevent bugs, it would be better to use python 3.7

Notes:
- It is highly recommended to upgrade numpy package before installing ttcrpy.

## Callable functions

6 functions can be called and run in this package:

- jntHypoVel_T : Joint hypocenter-velocity inversion of P wave data,
  parametrized via the velocity model.

- jntHyposlow_T : Joint hypocenter-velocity inversion of P wave data,
  parametrized via the slowness model.

- jntHypoVelPS_T : Joint hypocenter-velocity inversion of P- and S-wave data,
  parametrized via the velocity models.

- jntHyposlowPS_T : Joint hypocenter-velocity inversion of P- and S-wave data,
  parametrized via the slowness models.

- jointHypoVel_T : Joint hypocenter-velocity inversion of P wave data.
 Input data and inversion parameters are downloaded automatically from
 external text files.

- jointHypoVelPS_T : Joint hypocenter-velocity inversion of P- and S-wave data.
 Input data and inversion parameters are downloaded automatically
 from external text files.

## Examples

Two examples of hypocenter relocation using the JHVIT are presented (see : ).
The first example involves the inversion of P wave data while the second uses
both P- and S-wave data.

## References
```

@Thesis{Block91,
  Title                    = {Joint hypocenter-velocity inversion of local earthquake arrival
                              time data in two geothermal regions},
  Author                   = {Lisa Vectoria Block},
  Year                     = {1991},
  Number of Pages          = {448},
  University               = {Massachusetts Institute of Technology},
  Thesis Type              = {Doctoral thesis},
  keywords                 = {Earth, Atmospheric, and Planetary Sciences},
  URL                      = {http://hdl.handle.net/1721.1/13904}
}

@article{Block94,
  author                   = {Lisa V. Block, C. H. Cheng, Michael C. Fehler, and
                              W. Scott Phillips},
  title                    = {Seismic imaging using microearthquakes induced by hydraulic
                              fracturing},
  journal                  = {Geophysics},
  year                     = {1994},
  volume                   = {59},
  pages                    = {102-112},
  number                   = {1},
  doi                      = {10.1190/geo1992-0156},
  url                      = {https://library.seg.org/doi/10.1190/geo1992-0156}
}

@Thesis{Giroux01,
  Title                    = {Auscultation des barrages en béton par écoute microsismique:
                              détectabilité et localisation des événements},
  Author                   = {Bernard Giroux},
  Year                     = {2001},
  Number of Pages          = {268},
  University               = {Université de Montréal},
  Thesis Type              = {Doctoral thesis},
  Language                 = {French},
  URL                      = {https://publications.polymtl.ca/8641/}
}

@article{Nasr18,
  author                   = {Nasr, Maher and Giroux, Bernard and Dupuis, J. Christian},
  title                    = {A hybrid approach to compute seismic travel times in
                              three-dimensional tetrahedral meshes},
  journal                  = {Geophysical Prospecting},
  volume                   = {68},
  number                   = {4},
  pages                    = {1291-1313},
  keywords                 = {Travel time, Seismic modelling, Ray tracing, Seismics,
                              Computing aspects},
  doi                      = {10.1111/1365-2478.12930},
  url                      = {https://onlinelibrary.wiley.com/doi/abs/10.1111/1365-2478.12930},
  eprint                   = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/1365-2478.12930},
}

@article{Nasr21,
  author                   = {Nasr, Maher and Giroux, Bernard and Dupuis, J. Christian},
  title                    = {Python package for 3D joint hypocenter-velocity inversion on
                              tetrahedral meshes: Parallel implementation and practical
                              considerations},
  journal                  = {Computational Geosciences},
  volume                   = {n/a},
  number                   = {n/a},
  pages                    = {n/a},
  keywords                 = {joint hypocenter velocity inversion; hypocenter location;
                              passive seismic; computational seismology; parallelism;
                              tetrahedral meshes},
}

```
