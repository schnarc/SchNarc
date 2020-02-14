# SchNarc

A SchNetPack/SHARC interface for machine-learning accelerated excited state simulations.

# Installation requirements

You need a python installation with version 3.5 or later.
We recommend installing Anaconda with python 3 (see https://www.anaconda.com/distribution/).
If a package that you need, cannot be found, you can add channels (in this example conda-forge) with:
conda config --append channels conda-forge
However, it should not be required.
It is recommended to create an environment (in this example the environment is called ml) with:
conda create -n ml python h5py tensorboardX ase numpy pytorch six protobuf scipy matplotlib python-dateutil pyparsing kiwisolver cycler
Then activate the environment:
conda activate ml
If you haven't done so, get the SchNarc sources:
git clone https://github.com/schnarc/schnarc.git
Then go to the directory schnarc
cd schnarc
and carry out:
python setup.py --verbose build
Decide for an installation path, e.g., <some-path>/schnarc and inquire which python version you have installed with
python --version
The output is e.g. Python 3.7.6
Then add the following line to your .bashrc (where <version> would be 3.7 with the above example):
export PYTHONPATH=$PYTHONPATH:<some-path>/schnarc/lib/python<version>/site-packages/
Reload the .bashrc with
. ~/.bashrc
Then run
python setup.py --verbose install --prefix <some-path>/schnarc

Then install SHARC with pysharc (see https://sharc-md.org/?page_id=50#tth_sEc2.3 )

Training or running works in the same way, SchNet works, have a look at https://github.com/atomistic-machine-learning/schnetpack
