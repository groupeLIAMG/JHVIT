.. JHVIT documentation master file, created by
   sphinx-quickstart on Tue Oct 19 08:03:34 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##################################
Welcome to JHVIT's documentation!
##################################

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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started.rst
   example.rst
   results.rst
   code.rst
   references.rst


##################
Indices and tables
##################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
