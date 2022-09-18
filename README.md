# SchNarc

A SchNetPack/SHARC interface for machine-learning accelerated excited state simulations.

# System requirements

A personal computer (suggested RAM: 4GB or larger; suggested CPU: 1 core or more, 2.7 GHz or faster), ideally with a dedicated graphics card (GPU), is needed for installation running Linux. We have tested the software on different Linux flavors, e.g., Ubuntu 20.04 and RedHat 8.4.

# Installation and installation requirements

Many roads lead to Rome and there are many ways to install programs under linux, especially when using different variants of compiler optimizations. The following is a simplistic route to installing SchNarc. The installation when following the procedure outlined below is expected to take about 20-60 min on a "normal" desktop computer.

### Python and libraries

You need a python installation with version 3.5 or later.  
We recommend installing Miniconda with python 3 (see https://docs.conda.io/en/latest/miniconda.html) and mamba (see https://github.com/mamba-org/mamba).
Once you have miniconda installed, mamba is installed via
``conda install mamba -n base -c conda-forge``
If a package that you need, cannot be found, you can use different channels with the option -c or add channels (in this example conda-forge) with:  
``conda config --append channels conda-forge``  
It is recommended to create an environment (in this example the environment is called ml) with:  
*Note: Leave out the ``gfortran_linux-64 gcc_impl_linux-64 sysroot_linux-64=2.17`` and the commenting of gcc and gfortran below if you have a reasonably up-to-date gfortran compiler installed*  
``mamba create -n ml python h5py tensorboardX pytorch ase numpy six protobuf scipy matplotlib python-dateutil pyyaml tqdm pyparsing kiwisolver cycler netcdf4 hdf5 h5utils jupyter gfortran_linux-64 gcc_impl_linux-64 sysroot_linux-64=2.17``  
For some tasks, it is useful to install openbabel:  
``mamba install -n ml -c openbabel openbabel``  
Then activate the environment:  
``conda activate ml``   

### SHARC and pySHARC

Install SHARC with pysharc (see https://sharc-md.org/?page_id=50#tth_sEc2.3 or follow the instructions below; version 2.1.1) in a suitable folder
(``<some-path>``; inside this folder, git will create automatically a folder called sharc):  
``cd <some-path>``  
``git clone https://github.com/sharc-md/sharc.git``  
``cd sharc/source``  
Edit Makefile and make the following changes:  
``    USE_PYSHARC := true``  
``    USE_LIBS := mkl``  
``    ANACONDA := <path/to/anaconda>/anaconda3/envs/ml``  
``    MKLROOT := ${ANACONDA}/lib``  
``    #CC :=gcc``   (<- not of you have gcc and gfortran)  
``    #F90 :=gfortran``   (<- not of you have gcc and gfortran)  
``  LD= -L$(MKLROOT) -lmkl_rt -lpthread -lm -lgfortran $(NETCDF_LIB)``  
i.e., delete ``/lib/intel64`` after ``-L$(MKLROOT)``. The 4th change is a line that needs to be added (about MKLROOT). The 5rd and 6th change mean that you have to comment out the definition of CC and F90 and rather use the CC and F90 variables provided by the environment, which is set by anaconda to something like ``<your-anaconda-path>/x86_64-conda_cos6-linux-gnu-cc`` instead of gcc.  

Go to the pysharc/sharc folder:
``cd ../pysharc/sharc`` 
Edit \_\_init\_\_.py  there and make the following changes:  
``    #import sharc as sharc``

Go to the pysharc/netcdf folder:  
``cd ../netcdf``  
Edit Makefile  there and make the following changes:  
``    ANACONDA := <path/to/anaconda>/anaconda3/envs/ml``  
``    #CC=gcc``  (<- not of you have gcc and gfortran)  
Then go to the pysharc folder and run the installation procedure:  
``cd ..``  
``make install``  
Afterwards, go to the source folder and run the installation procedure:  
``cd ../source``  
``make install``  
As last step goto pysharc/lib  
``cd ../pysharc/lib``  
``cp libsharc.so <path/to/anaconda>/anaconda3/envs/ml/lib``  
then go to pysharc/sharc  
``cd ../sharc``  
``cp sharc.cpython-...-gnu.so <path/to/anaconda>/anaconda3/envs/ml/lib/sharc.so``  

### SchNet

Install SchNet in a suitable folder:  
``cd <some-path>``  
``git clone https://github.com/atomistic-machine-learning/schnetpack.git``  
Then go to the directory schnetpack  
``cd schnetpack``  
and carry out:  
``pip install .`` 

### SchNarc

If you haven't done so, get the SchNarc sources:  
``git clone https://github.com/schnarc/schnarc.git``  
Then go to the directory schnarc  
``cd schnarc``  
and carry out:  
``pip install .``  

Training or running works in the same way, SchNet works, have a look at https://github.com/atomistic-machine-learning/schnetpack or check-out the devel branch with a tutorial. The tutorial might take you approximately 2 or 3 hours.

## Troubleshooting

If your python version cannot find sharc, you can try the following:

Got to the pysharc/sharc folder
Edit \_\_init\_\_.py there and make the following changes:  
``  from . import sharc ``
and try to reinstall.

Make sure you have a  "sharc.cpython-...-gnu.so" file in the same folder after installation.  
  
Before you run the dynamics with sharc, source the sharcvars.sh file in the folder "sharc/bin/".  
  
If you compiled sharc with compilers in your conda environment, make sure to add the anaconda path to your ``$LD_LIBRARY_PATH``  
``LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path/to/anaconda>/anaconda3/envs/ml/lib``
