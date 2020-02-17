# SchNarc

A SchNetPack/SHARC interface for machine-learning accelerated excited state simulations.

# Installation requirements

You need a python installation with version 3.5 or later.  
We recommend installing Anaconda with python 3 (see https://www.anaconda.com/distribution/).  
If a package that you need, cannot be found, you can use different channels with the option -c (see below) or add channels (in this example conda-forge) with:  
``conda config --append channels conda-forge``  
It is recommended to create an environment (in this example the environment is called ml) with:  
``conda create -n ml python h5py tensorboardX pytorch ase numpy six protobuf scipy matplotlib python-dateutil pyyaml tqdm pyparsing kiwisolver cycler netcdf4 hdf5 h5utils``  
Then activate the environment:  
``conda activate ml``  
Then install further packages (pay attention to use a newer gcc version):  
``conda install -n ml -c msarahan gcc_linux-64``  
``conda install -n ml -c conda-forge lapack fftw``  

Install SHARC with pysharc (see https://sharc-md.org/?page_id=50#tth_sEc2.3 or follow the instructions below) in a suitable folder
(``<some-path>``; Inside this folder, git will create automatically a folder called sharc):
``cd <some-path>``  
``git clone https://github.com/sharc-md/sharc.git``  
``cd sharc/source``  
Edit Makefile and make the following changes:  
``    USE_PYSHARC := true``  
``    ANACONDA := <path/to/anaconda>/anaconda3/envs/pysharc``  
Go to the pysharc/netcdf folder:  
``cd ../pysharc/netcdf``  
Edit Makefile  there and make the following changes:  
``    ANACONDA := <path/to/anaconda>/anaconda3/envs/pysharc``  
Then go to the pysharc folder and run the installation procedure:  
``cd ..``  
``make install``  
Afterwards, go to the source folder and run the installation procedure:  
``cd ../source``  
``make install``  

Install SchNet in a suitable folder:  
``cd <some-path>``  
``git clone https://github.com/atomistic-machine-learning/schnetpack.git``  
Then go to the directory schnetpack  
``cd schnetpack``  
and carry out:  
``pip install .`` 

If you haven't done so, get the SchNarc sources:  
``git clone https://github.com/schnarc/schnarc.git``  
Then go to the directory schnarc  
``cd schnarc``  
and carry out:  
``pip install .``  

Training or running works in the same way, SchNet works, have a look at https://github.com/atomistic-machine-learning/schnetpack
