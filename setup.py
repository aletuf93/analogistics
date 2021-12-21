# -*- coding: utf-8 -*-
# setuptools references https://docs.python.org/3/distutils/setupscript.html
# pip references https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56


# first export requirements.txt file by using
# pip list --format=freeze > requirements.txt

# then upload to pypi using
# python setup.py sdist
# twine upload dist/*

from setuptools import setup, find_packages
from pathlib import Path
'''
import os
# read the requirements file
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []  # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()
'''

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='analogistics',
    version='0.2.0',
    description='Analytical Tools for Logistics System Design and Operations Management',
    long_description=long_description,
    url='https://github.com/aletuf93/analogistics',
    author='Alessandro Tufano',
    author_email='aletuf@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires='>= 3.6',
    install_requires=['osmnx',  # 1.1.1
                      'matplotlib',  # 3.2.2
                      'plotly',  # 5.4.0
                      'pandas',  # 0.25.3
                      'numpy',  # 1.19.5
                      'scipy',  # 1.5.3
                      'networkx',  # 2.5.1
                      'scikit-learn',  # 0.24.2
                      'seaborn',  # 0.11.0
                      'statsmodels',  # 0.12.1
                      'geocoder'],  # 1.38.1
    download_url='https://github.com/aletuf93/analogistics/archive/refs/tags/v0.1.tar.gz',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
    ],
)
