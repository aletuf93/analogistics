# -*- coding: utf-8 -*-
# setuptools references https://docs.python.org/3/distutils/setupscript.html
# pip references https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56


# first export requirements.txt file by using
# conda list -e > requirements.txt

from setuptools import setup
import os

# read the requirements file
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []  # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='analogistics',
    version='0.1.1',
    description='Analytical Tools for Logistics System Design and Operations Management',
    url='https://github.com/aletuf93/analogistics',
    author='Alessandro Tufano',
    author_email='aletuf@gmail.com',
    license='MIT',
    packages=['analogistics'],
    install_requires=install_requires,
    download_url='https://github.com/aletuf93/analogistics/archive/refs/tags/v0.1.tar.gz',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
    ],
)
