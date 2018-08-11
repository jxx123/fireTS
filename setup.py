from setuptools import setup
import os

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    dependencies = ['numpy', 'scipy', 'scikit-learn']
else:
    dependencies = [
        'numpy',
        'scikit-learn',
    ]

setup(
    name='fireTS',
    version='0.0.5',
    description='A python package for multi-variate time series prediction',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/jxx123/fireTS.git',
    author='Jinyu Xie',
    author_email='xjygr08@gmail.com',
    license='MIT',
    packages=['fireTS'],
    install_requires=dependencies,
    include_package_data=True,
    zip_safe=False)
