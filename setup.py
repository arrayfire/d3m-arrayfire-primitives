from setuptools import setup, find_packages
import os
import re
import sys

PACKAGE_NAME = 'af_primitives'
MINIMUM_PYTHON_VERSION = 3, 6

REQUIREMENTS_MAP = {}
def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))

def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert False, "'{0}' not found in '{1}'".format(key, module_path)

def package_from_requirement(requirement):
    """Convert pip requirement string to a package name."""
    return re.sub(r'-',
                  r'_',
                  re.sub(r'\[.*?\]|.*/([^@/]*?)(\.git)?.*',
                         r'\1',
                         requirement))

def read_requirements():
    """Read the requirements."""
    with open('requirements.txt') as requirements:
        return [package_from_requirement(requirement.strip())
                for requirement in requirements]

check_python_version()

setup(
    name=PACKAGE_NAME,
    version=read_package_variable('__version__'),
    description='ArrayFire D3M primitives',
    author='ArrayFire',
    keywords='d3m_primitive',
    packages=find_packages('af_primitives'),
    install_requires=read_requirements(),
    url='https://gitlab.datadrivendiscovery.org/syurkevitch/d3m-af-primitives/',
    entry_points = {
        'd3m.primitives': [
            'classification.k_neighbors.AFPrimitives = af_primitives.af_KNClassifier:af_KNClassifier',
            'regression.logistic.AFPrimitives = af_primitives.af_LogisticRegression:af_LogisticRegression'
        ],
    },
)
