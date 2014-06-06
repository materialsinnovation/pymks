#!/usr/bin/env python

from setuptools import setup, find_packages
setup(name='pymks',
      version='0.1-dev',
      description='Package for Materials Knowledge System (MKS) regression tutorial',
      author='Daniel Wheeler',
      author_email='daniel.wheeler2@gmail.com',
      url='http://wd15.github.com/pymks',
      packages=find_packages(),
      package_data = {'' : ['tests/*.py']}
      )
