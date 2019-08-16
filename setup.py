import os
import re
from setuptools import setup, find_packages

PACKAGE_NAME = 'af_primitives'

def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    raise KeyError("'{0}' not found in '{1}'".format(key, module_path))


setup(
    name=PACKAGE_NAME,
    version=read_package_variable('__version__'),
    description='ArrayFire Primitives',
    author=read_package_variable('__author__'),
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'arrayfire==3.6.20181017',
        'd3m==2019.6.7'
    ],
    url='https://gitlab.com/syurkevi/d3m-arrayfire-primitives',
    keywords='d3m_primitive',
    entry_points={
        'd3m.primitives': [
            'classification.logistic_regression.ArrayFire = af_primitives.af_LogisticRegression:af_LogisticRegression'
        ],
    },
)
