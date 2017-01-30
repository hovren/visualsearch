from distutils.core import setup

import os

scripts = [os.path.join('scripts', f) for f in os.listdir('./scripts/')]

setup(
    name='vsearch',
    version='1.0',
    packages=['vsearch'],
    url='http://users.isy.liu.se/cvl/hanov56/',
    license='GPLv3',
    author='Hannes Ovrén',
    author_email='hannes.ovren@liu.se',
    description='Visual search tool',
    include_package_data=True,
    scripts=scripts
)
