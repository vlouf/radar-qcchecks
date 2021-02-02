"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='radar_qcchecks',
    version='0.1',
    description='DP Radar Quality Checks.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/vlouf/radar-qcchecks',
    author='Valentin Louf',
    author_email='valentin.louf@bom.gov.au',
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='radar weather meteorology calibration',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['numpy', 'pandas', 'netCDF4', 'h5py', 'pyodim'],
    project_urls={
        'Bug Reports': 'https://github.com/vlouf/radar-qcchecks/issues',
        'Source': 'https://github.com/vlouf/radar-qcchecks/',
    },
)
