# SchNarc

This instruction shows how to install and use SchNarc for fitting dipole moment vectors and predicting UV/visible absorption spectra. The tutorial is based on Ref. 1 (https://aip.scitation.org/doi/10.1063/5.0021915)
A SchNetPack/SHARC interface for machine-learning accelerated excited state simulations.

# Installation and installation requirements

Many roads lead to Rome and there are many ways to install programs under linux, especially when using different variants of compiler optimizations. The following is a simplistic route to installing SchNarc.

You need a python installation with version 3.5 or later.  
We recommend installing Anaconda with python 3 (see https://www.anaconda.com/distribution/).  
If a package that you need, cannot be found, you can use different channels with the option -c or add channels (in this example conda-forge) with:  
``conda config --append channels conda-forge``  
It is recommended to create an environment (in this example the environment is called ml) with:
Note: Leave out the gfortran_linux-64 and the commenting of gcc and gfortran below if you have a reasonably up-to-date gfortran compiler installed
``mamba create -n ml python h5py tensorboardX pytorch ase numpy six protobuf scipy matplotlib python-dateutil pyyaml tqdm pyparsing kiwisolver cycler netcdf4 hdf5 h5utils jupyter gfortran_linux-64``
For some tasks, it is useful to install openbabel:
``mamba install -n ml -c openbabel openbabel``
Then activate the environment:
``conda activate ml``

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
Below, a short tutorial on how to get started will be provided.

You do not need to install SHARC for this purpose. However, if you wish to, you can install SHARC with pysharc (see https://sharc-md.org/?page_id=50#tth_sEc2.3 or follow the instructions below) in a suitable folder
(``<some-path>``; inside this folder, git will create automatically a folder called sharc):  
``cd <some-path>``  
``git clone https://github.com/sharc-md/sharc.git``  
``cd sharc/source``  
Edit Makefile and make the following changes:  
``    USE_PYSHARC := true``  
``    USE_LIBS := mkl``  
``    ANACONDA := <path/to/anaconda>/anaconda3/envs/ml``  
``    MKLROOT := ${ANACONDA}/lib``  
``  #CC :=gcc``  
``  #F90 :=gfortran``  
``  LD= -L$(MKLROOT) -lmkl_rt -lpthread -lm -lgfortran $(NETCDF_LIB)``  
i.e., delete ``/lib/intel64`` after ``-L$(MKLROOT)``. The 4th change is a line that needs to be added (about MKLROOT). The 5rd and 6th change mean that you have to comment out the definition of CC and F90 and rather use the CC and F90 variables provided by the environment, which is set by anaconda to something like ``<your-anaconda-path>/x86_64-conda_cos6-linux-gnu-cc`` instead of gcc.  

Got to the pysharc/sharc folder:
``cd ../pysharc/sharc`` 
Edit \_\_init\_\_.py  there and make the following changes:  
``    #import sharc as sharc``

Go to the pysharc/netcdf folder:  
``cd ../netcdf``  
Edit Makefile  there and make the following changes:  
``    ANACONDA := <path/to/anaconda>/anaconda3/envs/pysharc``  
``    #CC=$(ANACONDA)/bin/gcc``  
Then go to the pysharc folder and run the installation procedure:  
``cd ..``  
``make install``  
Afterwards, go to the source folder and run the installation procedure:  
``cd ../source``  
``make install``  
# Troubleshooting

If your python version cannot find sharc, you can try the following:

Got to the pysharc/sharc folder
Edit \_\_init\_\_.py there and make the following changes:  
``  from . import sharc ``
and try to reinstall.

Make sure you have a  "sharc.cpython-...-gnu.so" file in the same folder after installation. 

Before you run the dynamics with sharc, source the sharcvars.sh file in the folder "sharc/bin/".

# Getting started

The data sets used for this short tutorial can be found at figshare: doi:10.6084/m9.figshare.14832396 
The trained model is provided in the folder "Model". Datasets to train the models are in the folder "Datasets" and InitialConditions for UV/visible absorption spectra are in the folder "InitialConditions". They are already saved in an ase data base format (see: https://wiki.fysik.dtu.dk/ase/tutorials/tutorials.html if you are unfamiliar with ase). 

## Training, validating, and testing

To train an ML model, have a look at the Test-folder provided in the examples-folder. 
Different from SchNet, you need a "tradeoffs.yaml" file, which contains the properties you want to train on, the tradeoff to weigh this property and the number of singlets, doublets, triplets, and quartet states you want to train.

In the "examples" folder, you will find a file called "run.sh". You have to specify your train-folder, the path to your data base and the path to the SchNarc scripts. The first execution shows how to train a model, the second how to validate it (possible splits for the evalzation mode are "all", "train", "validation", and "test"). The evaluation file is written into your training folder and is called "evaluation.csv". The last execution shows you how to predict molecules in a DB-file. In the example, you will predict the whole training set.
The arguments used for splitting, the batch size and some other model parameters are identical to SchNet. Check the SchNet-tutorial if you want to know about hyperparameter optimizations.



# Tutorial instructions for the chapter "Learning Excited State Properties" in the book "Quantum Chemistry in the Age of Machine Learning" edited by Pavlo Dral

After downloading the files from figshare (https://doi.org/10.6084/m9.figshare.13635353), you can use the models provided in the "MLModels" folder to predict the energies and dipole moments for the inital conditions provided in the InitialConditions folder. 
You will need to install matplotlib to your anaconda version: ``conda install -c conda-forces matplotlib``
Additional instructions and README-files are provided in each of the zipped folders.

## Absorption spectrum of CH2NH2+
The first task is to predict the spectrum of CH2NH2+ with the model trained on this molecule. Therefore, execute the following:

``python $SCHNARC/run_schnarc.py pred InitialConditions/CH2NH2_InitialConditions.db MLModels/CH2NH2+/`` 

You use the prediction mode of SchNarc to predict all data points in the db file "CH2NH2\_InitialConditions.db" using the model provided in the folder "MLModels/CH2NH2+/". The predictions will be saved in the model folder.
Make a new folder named "CH2NH2+\_spectrum" and copy the "predictions.npz" file into this folder. Copy the "Spectrum.ipynb" file from the "examples" folder into this folder.
Open the jupyter notebook "Spectrum.ipynb" and go through the different steps to get your absorption spectrum.
By changing the datapath in the command you used before and the ML model, you can predict spectra with different models and for different molecules. We recommend you make a separate folder for each of the different predictions.

## Getting the charges and electrostatic potentials

Now copy the jupyter notebook "Charges.ipynb" from the "examples" folder and go through the steps. You do not have to make a separate prediction, the charges are already saved in the file "predictions.npz" from the previous exercise.
To plot the electrostatic potentials, you need open babel and jmol.
