#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

__minimum_numpy_version__ = '1.10.0'
__minimum_tensorflow_version__ = '1.8.0'

setup_requires = ['numpy>=' + __minimum_numpy_version__, 
'tensorflow>='+__minimum_tensorflow_version__]

setup(name='bayes_filter',
      version='0.0.1',
      description='Bayesian method for directional and temporal TEC model of wrapped phase screens',
      author=['Josh Albert'],
      author_email=['albert@strw.leidenuniv.nl'],
    setup_requires=setup_requires,  
    tests_require=[
        'pytest>=2.8',
    ],
    package_data= {'bayes_filter':['arrays/*']},
   package_dir = {'':'./'},
   packages=find_packages('./')
     )

