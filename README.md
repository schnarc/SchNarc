# SchNarc

A SchNetPack/SHARC interface for machine-learning accelerated excited state simulations.

# Installation requirements

Install SHARC with pysharc (see https://sharc-md.org/?page_id=50#tth_sEc2.3 )

You need a python installation with version 3.5 or later.  
We recommend installing Anaconda with python 3 (see https://www.anaconda.com/distribution/).  
If a package that you need, cannot be found, you can add channels (in this example conda-forge) with:  
``conda config --append channels conda-forge``  
It is recommended to create an environment (in this example the environment is called ml) with:  
``conda create -n ml python h5py tensorboardX ase numpy torch six protobuf scipy matplotlib python-dateutil pyparsing kiwisolver cycler``  
Then activate the environment:  
``conda activate ml``  

Install SchNet in a suitable folder:  
``cd <parent-folder-where-schnetpack-folder-will-be-created-automatically>``  
``git clone https://github.com/atomistic-machine-learning/schnetpack.git``  
Then go to the directory schnetpack  
``cd schnetpack``  
and carry out:  
``python setup.py --verbose build``  
Installation can take place in the same directory, e.g., ``<some-path>/schnetpack ``  
Therefore run  
``python setup.py --verbose install --prefix <some-path>/schnetpack``  
You will be told to adjust your PYTHONPATH  
Do as you are told e.g. by  
``export PYTHONPATH=$PYTHONPATH:<some-path>/schnetpack/lib/python<version>/site-packages/``  
Reload the .bashrc with  
``. ~/.bashrc``  
Then re-run  
``python setup.py --verbose install --prefix <some-path>/schnetpack``  


If you haven't done so, get the SchNarc sources:  
``git clone https://github.com/schnarc/schnarc.git``  
Then go to the directory schnarc  
``cd schnarc``  
and carry out:  
``python setup.py --verbose build``  
Installation can take place in the same directory, e.g., ``<some-path>/schnarc``   
Therefore run  
``python setup.py --verbose install --prefix <some-path>/schnarc``  
You will be told to adjust your PYTHONPATH  
Do as you are told e.g. by  
``export PYTHONPATH=$PYTHONPATH:<some-path>/schnarc/lib/python<version>/site-packages/``  
Reload the .bashrc with  
``. ~/.bashrc``  
Then re-run  
``python setup.py --verbose install --prefix <some-path>/schnarc``  

Training or running works in the same way, SchNet works, have a look at https://github.com/atomistic-machine-learning/schnetpack
