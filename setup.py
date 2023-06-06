# from setuptools import setup
# setup(name='spectra_par_estimation',
# version='0.1',
# description='Testing installation of spectra package',
# url='#',
# author='auth',
# author_email='me18b106@smail.iitm.ac.in',
# license='MIT',
# packages=['spectra_par_estimation'],
# zip_safe=False)


import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


requirements = [
    # use environment.yml
]


setup(
    name="kinetic_parameter_estimation",
    version="0.0.1",
    url="#",
    author="Varshith",
    author_email="me18b106@smail.iitm.ac.in",
    description="Python package for Kinetic parameter estimation",
    long_description=read("README.rst"),
    packages=find_packages(exclude=("tests",)),
    # entry_points={
    #     "console_scripts": [
    #         "kinetic_parameter_estimation=kinetic_parameter_estimation.cli:cli"
    #     ]
    # },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ],
)
