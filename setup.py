# -*- coding: utf-8 -*-
# TODO: look here for references https://docs.python.org/3/distutils/setupscript.html
from setuptools import setup

setup(
    name='analogistics',
    version='0.1.0',
    description='Analytical Tools for Logistics System Design and Operations Management',
    url='https://github.com/aletuf93/analogistics',
    author='Alessandro Tufano',
    author_email='aletuf@gmail.com',
    license='MIT',
    packages=['pyexample'],
    install_requires=['mpi4py>=2.0',
                      'numpy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
