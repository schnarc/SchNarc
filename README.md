# SchNarc
A SchNetPack/SHARC interface for machine-learning accelerated excited state simulations.

This instruction shows how to install and use SchNarc for fitting dipole moment vectors and predicting UV/visible absorption spectra. The tutorial is based on Ref. 1 (https://aip.scitation.org/doi/10.1063/5.0021915)


# System requirements
A personal computer (suggested RAM: 4GB or larger; suggested CPU: 1 core or more, 2.7 GHz or faster), ideally with a dedicated graphics card (GPU), is needed for installation running Linux. We have tested the software on different Linux flavors, e.g., Ubuntu 20.04 and RedHat 8.4.

# Installation and installation requirements
Many roads lead to Rome and there are many ways to install programs under linux, especially when using different variants of compiler optimizations. The following is a simplistic route to installing SchNarc. The installation when following the procedure outlined below is expected to take about 20-60 min on a "normal" desktop computer.

## Python and libraries
You need a python installation with version 3.5 or later.
We recommend installing Miniconda with python 3 (see https://docs.conda.io/en/latest/miniconda.html) and mamba (see https://github.com/mamba-org/mamba). Once you have miniconda installed, mamba is installed via conda install 
``mamba -n base -c conda-forge`` 

If a package that you need, cannot be found, you can use different channels with the option -c or add channels (in this example conda-forge) with:
``conda config --append channels conda-forge`` 

It is recommended to create an environment (in this example the environment is called ml) with:
``mamba create -n ml python h5py tensorboardX pytorch ase numpy six protobuf scipy matplotlib python-dateutil pyyaml tqdm pyparsing kiwisolver cycler netcdf4 hdf5 h5utils jupyter gfortran_linux-64`` 

Note: Leave out the gfortran_linux-64 and the commenting of gcc and gfortran below if you have a reasonably up-to-date gfortran compiler installed

For some tasks, it is useful to install openbabel:

``mamba install -n ml -c openbabel openbabel`` 

Then activate the environment:
``conda activate ml``   

## SchNet
Install SchNet in a suitable folder:
``cd <some-path>``
``git clone https://github.com/atomistic-machine-learning/schnetpack.git`` 
Then go to the directory schnetpack
``cd schnetpack`` 
and carry out:
``pip install .`` 

## SchNarc
If you haven't done so, get the SchNarc sources:
``git clone https://github.com/schnarc/schnarc.git`` 
Then go to the directory schnarc
``cd schnarc`` 
and carry out:
``pip install .`` 
Aftwards, please change to the "DipoleMoments_Spectra" branch via:
``git checkout DipoleMoments_Spectra`` 



# Getting started

Training or running works in the same way, SchNet works, have a look at https://github.com/atomistic-machine-learning/schnetpack or check-out the devel branch for training models for excited-state dynamics or this branch and the file "Tutorial.md" to learn about training dipole moment vectors for UV/visible absorption spectra and electrostatic potentials. The tutorial might take you approximately 2 or 3 hours. Below is a very brief example on how to get started.

The data sets used for this short tutorial can be found at figshare: https://doi.org/10.6084/m9.figshare.14832396.v1
The trained model is provided in the folder "ML Models". Datasets to train the models are in the folder "Datasets" and InitialConditions for UV/visible absorption spectra are in the folder "InitialConditions". They are already saved in an ase data base format (see: https://wiki.fysik.dtu.dk/ase/tutorials/tutorials.html if you are unfamiliar with ase). If you are curious on learning how to train a model go to the next section, otherwise, skip it and directly go to the file "Tutorial.md".

## Training, validating, and testing

To train an ML model, have a look at the "Training" folder. 
Different from SchNet, you need a "tradeoffs.yaml" file, which contains the properties you want to train on, the tradeoff to weigh this property and the number of singlets, doublets, triplets, and quartet states you want to train.

In the "Training" folder, you will find two files called "train_cpu.sh" and "train_gpu.sh" that show how to train a model on a CPU and GPU, respectively. You have to specify your train-folder, the path to your data base and the path to the SchNarc scripts. The corresponding "eval_cpu.sh" and "eval_gpu.sh" files show how to validate the model (possible splits for the evalzation mode are "all", "train", "validation", and "test"). The evaluation file is written into your training folder and is called "evaluation.csv". The "pred_cpu.sh" and "pred_gpu.sh" files show executions to predict molecules in a DB-file. In the example, you will predict the whole training set.
The arguments used for splitting, the batch size and some other model parameters are identical to SchNet. Check the SchNet-tutorial if you want to know about hyperparameter optimizations.


