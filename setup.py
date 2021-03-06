#! /usr/bin/env python

# Author: Alexander Fabisch <afabisch@googlemail.com>
# License: BSD 3 clause

from setuptools import setup
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
            "test": ["nose", "coverage"]
        }
    )


if __name__ == "__main__":
    setup_package()
