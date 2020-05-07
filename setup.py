#!/usr/bin/env python

from distutils.core import setup

setup(name='grace_dl',
      version='1.0',
      description='GRAdient Compression Framework',
      url='https://github.com/sands-lab/grace',
      packages=['grace_dl'], requires=['horovod', 'torch', 'tensorflow']
      )
