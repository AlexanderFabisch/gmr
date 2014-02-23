#! /usr/bin/env python

# Author: Alexander Fabisch <afabisch@googlemail.com>
# License: BSD 3 clause


from distutils.core import setup


def setup_package():
    setup(
        name="gmr",
        version="1.0",
        author="Alexander Fabisch",
        author_email="afabisch@googlemail.com",
        description="Gaussian Mixture Regression",
        long_description="Gaussian Mixture Models (GMMs) for clustering and "
                         "regression.",
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
            "Programming Language :: Python :: 3.3",
        ],
        packages=["gmr"],
        requires=["numpy", "sklearn"],
    )


if __name__ == "__main__":
    setup_package()
