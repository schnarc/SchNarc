import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8') as f:
        return f.read()

setup(
    name='schnarc',
    version='0.0.0',
    author='Michael Gastegger, Julia Westermayr, Philipp Marquetand',
    email='michael.gastegger@tu-berlin.de',
    packages=find_packages('src'),
    scripts=['src/scripts/schnarc_md.py', 'src/scripts/run_schnarc.py', 'src/scripts/setup_schnarc.py', 'src/scripts/transform_prediction_QMout.py', 'src/scripts/countstep.py', 'src/scripts/get_xyzandproperties_deltaE.py', 'src/scripts/QM2output.py', 'src/scripts/write_geoms.py','src/scripts/transform_dataset.py'],
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
