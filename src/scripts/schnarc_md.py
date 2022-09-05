import sys
import os
import schnarc
import logging
import shutil
import argparse
import numpy as np
import datetime
import copy
import time
(tc, tt) = (time.process_time(), time.time())
#import schnet
import schnetpack as spk
import torch
import torch.nn as nn
from ase.units import Hartree, Bohr, Debye, invcm
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler
from sharc.pysharc.interface import SHARC_INTERFACE
import run_schnarc as rs
#import sharc and schnarc
from functools import partial
from schnarc.calculators import SchNarculator, EnsembleSchNarculator

def _load_model(modelpath,device):
    if os.path.isdir(modelpath):
        modelpath = os.path.join(modelpath, 'best_model')

    if not torch.cuda.is_available():
        model = torch.load(modelpath, map_location='cpu')
    else:
        model = torch.load(modelpath)
    # Check if parallel
    model = model.to(device)
    return model

def save_model(modelpath,model):
    if os.path.isdir(modelpath):
        modelpath = os.path.join(modelpath,"best_model")
    model = torch.save(model,modelpath)

class SHARC_NN(SHARC_INTERFACE):
    """
    Class for SHARC NN
    """
    # Name of the interface
    interface = 'NN'
    # store atom ids
    save_atids = True
    # store atom names
    save_atnames = True
    # accepted units:  0 : Bohr, 1 : Angstrom
    iunit = 0
    # not supported keys
    not_supported = ['nacdt', 'dmdr' ]

    def initial_setup(self, **kwargs):
        return

    def do_qm_job(self, tasks, Crd):
        """

        Here you should perform all your qm calculations

        depending on the tasks, that were asked
        Hamiltonian matrix and dipole moment matrix are complex
        Gradient and NACs are real
        """
        QMout = self.schnarc_init.calculate(Crd)
        #print("Compare prediction:")
        #print(QMout)
        return QMout

    def final_print(self):
        self.sharc_writeQMin()

    def parseTasks(self, tasks):
        """
        these things should be interface dependent

        so write what you love, it covers basically everything
        after savedir information in QMin

        """

        # find init, samestep, restart
        QMin = { key : value for key, value in self.QMin.items() }
        QMin['natom'] = self.NAtoms
        QMin['atname'] = self.AtNames
        self.n_atoms=self.NAtoms
        key_tasks = tasks['tasks'].lower().split()


        if any( [self.not_supported in key_tasks ] ):
            print( "not supported keys: ", self.not_supported )
            sys.exit(16)

        for key in key_tasks:
            QMin[key] = []

        for key in self.states:
            QMin[key] = self.states[key]


        if 'init' in QMin:
            checkscratch(QMin['savedir'])
        if not 'init' in QMin and not 'samestep' in QMin and not 'restart' in QMin:
            fromfile=os.path.join(QMin['savedir'],'U.out')
            if not os.path.isfile(fromfile):
                print( 'ERROR: savedir does not contain U.out! Maybe you need to add "init" to QM.in.' )
                sys.exit(1)
            tofile=os.path.join(QMin['savedir'],'Uold.out')
            shutil.copy(fromfile,tofile)

        for key in ['grad', 'nacdr']:
            if tasks[key].strip() != "":
                QMin[key] = []

        QMin['pwd'] = os.getcwd()

        return QMin

    def readParameter(self, param,  *args, **kwargs):

        # Get device
        self.device = torch.device("cuda" if param.cuda else "cpu")
        self.NAtoms = vars(self)['NAtoms']
        self.AtNames = vars(self)['AtNames']
        self.dummy_crd = np.zeros((self.NAtoms,3))
        self.hessian = True if param.hessian else False
        self.adaptive = param.adaptive
        self.socs_mask = param.socs_mask
        self.finish = param.finish
        #self.nac_approx=param.nac_approx
        #self.thresholds = param.thresholds
        # get thresholds for adaptive sampling for the different properties
        self.thresholds = {}
        self.thresholds['energy'] = param.thresholds[0]#/Hartree
        self.thresholds['gradients'] = param.thresholds[1]#/Hartree/Bohr
        self.thresholds['forces'] = param.thresholds[1]
        self.thresholds['dipoles'] = param.thresholds[2]#/Debye
        self.thresholds['nacs'] = param.thresholds[3]
        self.thresholds['socs'] = param.thresholds[4]#/invcm
        self.thresholds['diab'] = np.inf
        self.thresholds['diab2'] = np.inf
        if param.nac_approx is not None:
            self.nac_approx=param.nac_approx
        else:
            self.nac_approx = None
        if self.adaptive is not None:
            self.NNnumber = int(2)
            self.modelpaths=param.modelpaths
            self.modelpaths.append(param.modelpath)
            #self.modelpaths.append(param.modelpaths[:])
            self.schnarc_init = EnsembleSchNarculator(self.dummy_crd,self.AtNames,self.modelpaths,param,hessian=self.hessian,nac_approx=self.nac_approx,adaptive=self.adaptive,thresholds=self.thresholds,print_uncertainty=param.print_uncertainty)
        else:
            self.NNnumber = int(1) #self.options["NNnumber"]
            self.schnarc_init = SchNarculator(self.dummy_crd,self.AtNames,param.modelpath,param,hessian=self.hessian,nac_approx=self.nac_approx,adaptive=self.adaptive,thresholds=self.thresholds,print_uncertainty=param.print_uncertainty)
        return


def main():
    """
        Main Function if program is called as standalone
    """
    parser = rs.get_parser()
    args = parser.parse_args()
    print( "Initialize: CPU time:% .3f s, wall time: %.3f s"%(time.process_time() - tc, time.time() - tt))
    if args.adaptive is not None:
        adaptive = args.adaptive
    else:
        adaptive = None
    dataset = spk.data.AtomsData(args.datapath, environment_provider=args.environment_provider,collect_triples="schnet"=='wacsf')
    metadata = dataset.get_metadata()
    n_singlets = metadata["n_singlets"]
    n_triplets = metadata["n_triplets"]
    all_states = n_singlets + 3 * n_triplets
    n_socs =  int(all_states * (all_states - 1))
    socs=False
    parallel = args.parallel
    for k in metadata.keys():
        if k=="socsindex":
            socindex=metadata[k]
            mask_socs = np.zeros((n_socs))
            for socindex_mask,maskvalue in enumerate(socindex):
                mask_socs[maskvalue]=1.0
            args.socs_mask=mask_socs
            socs=True
    if socs == False:
        args.socs_mask = np.ones((n_socs))
    if args.cuda == True:
        device = "cuda"
    else:
        device = "cpu"
    param = "schnarc_options"
    # init SHARC_NN class 
    nn = SHARC_NN()
    print( "init SHARC_NN:  CPU time: % .3f s, wall time: %.3f s"%(time.process_time() - tc, time.time() - tt))
    # run sharc dynamics
    nn.run_sharc("input",args,adaptive, initial_param=param)
    print( "run dynamics:  CPU time: % .3f s, wall time: %.3f s"%(time.process_time() - tc, time.time() - tt))
if __name__ == "__main__":
    main()



