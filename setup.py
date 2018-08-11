from setuptools import setup

setup(
    name='fireTS',
    version='0.0.4',
    description='A python package for multi-variate time series prediction',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/jxx123/fireTS.git',
    author='Jinyu Xie',
    author_email='xjygr08@gmail.com',
    license='MIT',
    packages=['fireTS'],
    install_requires=[
        'numpy',
        'scikit-learn',
    ],
    include_package_data=True,
    zip_safe=False)
