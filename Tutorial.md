
# Tutorial instructions for the chapter "Learning Excited State Properties" in the book "Quantum Chemistry in the Age of Machine Learning" edited by Pavlo Dral

After downloading the files from figshare (https://doi.org/10.6084/m9.figshare.14832396.v2), you can use the models provided in the "MLModels" folder to predict the energies and dipole moments for the inital conditions provided in the InitialConditions folder.
Additional instructions and README-files are provided in each of the zipped folders.

### Atention: Make sure you changed to the branch "DipoleMoments_Spectra" before moving on with this tutorial.

We recommend setting the SchNarc path via ``export SCHNARC="<your path>/schnarc/src/scripts/"``.

## Absorption spectrum of CH2NH2+
The first task is to predict the spectrum of CH2NH2+ with the model trained on this molecule. Therefore, execute the following:

``python $SCHNARC/run_schnarc.py pred InitialConditions/CH2NH2_InitialConditions.db MLModels/CH2NH2+/`` 

You use the prediction mode of SchNarc to predict all data points in the db file "CH2NH2\_InitialConditions.db" using the model provided in the folder "MLModels/CH2NH2+/". The predictions will be saved in the model folder.
Make a new folder named "CH2NH2+\_spectrum" and copy the "MLModels/CH2NH2+/predictions.npz" file into this folder. Copy the "Spectrum.ipynb" file from the "Tests/CH2NH2+\_spectrum/" folder into this folder.
Open the jupyter notebook "Spectrum.ipynb" and go through the different steps to get your absorption spectrum.
By changing the datapath in the command you used before and the ML model, you can predict spectra with different models and for different molecules. We recommend you make a separate folder for each of the different predictions.

## Getting the charges and electrostatic potentials

Now copy the jupyter notebook "Charges.ipynb" from the "Tests/CH2NH2+\_charges/"" folder and go through the steps. You do not have to make a separate prediction, the charges are already saved in the file "predictions.npz" from the previous exercise.
To plot the electrostatic potentials, you need open babel and jmol.
