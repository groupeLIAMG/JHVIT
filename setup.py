import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(

    name="JHVIT",
    version="0.0.9",
    author="Maher Nasr",
    author_email="Maher.Nasr@inrs.ca",
    description="Code to perform seismic hypocenter location on tetrahedral meshes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/groupeLIAMG/JHVIT",
    keywords = ["joint hypocenter velocity inversion", "hypocenter location",
                "passive seismic", "tetrahedral meshes"],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires= "==3.7",
    install_requires=["numpy>=1.20.1",
                      "scipy",
                      "vtk",
                      "ttcrpy>=1.1.8"],
)
