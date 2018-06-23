from setuptools import setup

setup(
    name='fireTS',
    version='0.0.1',
    description='A python package for time series prediction',
    url='https://github.com/jxx123/fireTS',
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
