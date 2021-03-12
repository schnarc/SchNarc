import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8') as f:
        return f.read()

setup(
    name='schnet_ev',
    version='0.0.0',
    author='Julia Westermayr, Michael Gastegger, Kristof Schütt, Reinhard J. Maurer',
    email='julia.westermayr@warwick.ac.uk',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.6',
    install_requires=[
        "torch>=0.4.1",
        "numpy",
        "ase>=3.16",
        "tensorboardX",
        "h5py"
    ]
)
