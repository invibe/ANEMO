#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from setuptools import setup, find_packages
 
import ANEMO

setup(
    name='ANEMO',
    version=ANEMO.__version__,
    packages=find_packages(),
    author="",
    author_email="",
    description="",
    long_description=open('README.md').read(),
    include_package_data=True,
    url='',
    classifiers=[
        "Programming Language :: Python",
    ],
)
