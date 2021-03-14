#! /usr/bin/env python

# Author: Alexander Fabisch <afabisch@googlemail.com>
# License: BSD 3 clause

from setuptools import setup

try:
    import builtins
except ImportError:  # Python 2
    import __builtin__ as builtins
# Idea from scikit-learn:
# We set a global variable so that the __init__ can detect if it is being
# loaded by the setup routine, to avoid attempting to load components that
# require additional dependencies that are not yet installed.
builtins.__GMR_SETUP__ = True

import gmr


def setup_package():
    setup(
        name="gmr",
        version=gmr.__version__,
        author="Alexander Fabisch",
        author_email="afabisch@googlemail.com",
        url="https://github.com/AlexanderFabisch/gmr",
        description="Gaussian Mixture Regression",
        long_description=open("README.rst").read(),
        license="new BSD",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        packages=["gmr"],
        install_requires=["numpy", "scipy"],
        extras_require={
            "all": ["matplotlib", "scikit-learn", "svgpathtools"],
            "test": ["nose", "coverage"],
            "doc": ["pdoc3"],
        }
    )


if __name__ == "__main__":
    setup_package()
