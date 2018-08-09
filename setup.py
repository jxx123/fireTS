from setuptools import setup

setup(
    name='fireTS',
    version='0.0.1',
    description='A python package for multi-variate time series prediction',
    long_description=open('README.md').read(),
    url='https://github.com/jxx123/fireTS.git',
    author='Jinyu Xie',
    author_email='xjygr08@gmail.com',
    license='MIT',
    packages=['fireTS'],
    install_requires=[
        'numpy',
        'sklearn',
    ],
    include_package_data=True,
    zip_safe=False)
