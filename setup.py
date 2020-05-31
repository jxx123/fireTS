from setuptools import setup

dependencies = [
    'numpy',
    'scipy',
    'scikit-learn',
]

setup(
    name='fireTS',
    version='0.0.8',
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
