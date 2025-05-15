import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="xhermes",
    version="0.1.0",
    description="Analyse data from Hermes-3 simulations using xarray",
    license="Apache",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=["xarray", "xbout"],
    packages=find_packages(),
    include_package_data=True,
)
