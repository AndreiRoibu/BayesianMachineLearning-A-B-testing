from setuptools import setup, find_packages

setup(
    name='Bayesian Machine Learning: A/B Testing',
    version='0.0.1',
    description='A series of basic logistic regression files',
    license='BSD 3-clause license',
    maintainer='AndreiRoibu',
    maintainer_email='aroibu1@gmail.com',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'sklearn',
        'seaborn',
        'sortedcontainers',
        'tornado',
        'requests',
        'ipykernel',
        'statsmodels',
        'flask',
    ],
)