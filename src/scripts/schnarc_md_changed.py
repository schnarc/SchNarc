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
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler
sharc_path = "/user/brigitta/sharc_main/DEVELOPMENT/pysharc/bin"
sys.path.append(sharc_path)
sys.path.append(os.path.join(sharc_path,'..','lib'))
sys.path.append(os.path.join(sharc_path,'..','lib', 'sharc'))
from sharc.pysharc.interface import SHARC_INTERFACE
import run_schnarc as rs
#import sharc and schnarc
from functools import partial
from schnarc.calculators import SchNarculator, EnsembleSchNarculator

class SHARC_NN(SHARC_INTERFACE):
    """
    Class for SHARC NN
    """
    # Analytical functions used
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
        QMout={}
        # translate Coordinates to xyz and initialise Variables
        x=Crd[0][0]
        y=Crd[0][1]
        z=Crd[0][2]
        k=0.1
        x1_S=-1.0
        x2_S=1.0
        E=0.0
        n=0.02
        # calculate hamiltonian
        Hamiltonian=np.zeros((2,2),dtype=complex)
        Hamiltonian[0][0]=complex(0.5*k*(x-(x1_S))**2+E+1.0*y**2+1.0*z**2 , 0.0000)
        Hamiltonian[1][1]=complex(0.5*k*(x-(x2_S))**2+E+1.0*y**2+1.0*z**2 , 0.0000)
        Hamiltonian[0][1]=complex(n, 0.0000)
        d,U=np.linalg.eigh(Hamiltonian)
        DHamiltonian=np.diag(d)
        QMout["h"]= np.array(DHamiltonian).tolist()
        # calculate gradiants per surface
        QMout["overlap"]=U
        natom=1
        nmstates=2
        SH2ANA={}
        SH2ANA["gvar"]=[["x","y","z"]]
        deriv={}
        deriv["x"]=[[complex(k*(x-(x1_S)) , 0.00),complex(0.00 , 0.00)],[complex(0.00 , 0.00),complex(k*(x-(x2_S)),0.00)]]
        deriv["y"]=[[complex(2*1.0*y , 0.0000),complex(0.0000 , 0.0000)],[complex(0.0000,0.0000),complex(2*1.0*y,0.0000)]]
        deriv["z"]=[[complex(2*1.0*z , 0.0000),complex(0.0000 , 0.0000)],[complex(0.0000,0.0000),complex(2*1.0*z,0.0000)]]
        QMout["phases"]=[ complex(1.,0.) for i in range(nmstates) ]
        for i in range(nmstates):
           if QMout['overlap'][i][i].real<0.:
              QMout['phases'][i]=complex(-1.,0.)
        SH2ANA["deriv"]=deriv
        ggrad=[]
        Ucon=[ [ U[i][j].conjugate() for i in range(len(U)) ] for j in range(len(U)) ]
        for iatom in range(natom):
           ggrad.append([])
           for idir in range(3):
              if SH2ANA['gvar'][iatom][idir]=='0':
                  ggrad[-1].append( [ 0. for i in range(nmstates) ] )
              else:
                  v=SH2ANA['gvar'][iatom][idir]
                  A=SH2ANA['deriv'][v]
                  Dmatrix=np.dot(Ucon,np.dot(A,U))
                  ggrad[-1].append( [ Dmatrix[i][i].real for i in range(nmstates) ] )
        # rearrange gradients
        grad=[ [ [ ggrad[iatom][idir][istate] for idir in range(3) ] for iatom in range(natom) ] for istate in range(nmstates) ]
        QMout["grad"]=grad
        # sharc needs dipole moments here only 0
        QMout["dm"]=np.array(np.zeros((3,2,2))).tolist()
        #QMout = self.schnarc_init.calculate(Crd)
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
        #self.nac_approx=param.nac_approx
        if param.thresholds is not None:
            self.thresholds = {}
            self.thresholds['energy'] = param.thresholds[0]
            self.thresholds['gradients'] = param.thresholds[1]
            self.thresholds['dipoles'] = param.thresholds[2]
            self.thresholds['nacs'] = param.thresholds[3]
            self.thresholds['socs'] = param.thresholds[4]
            self.thresholds['diab'] = np.inf
            self.thresholds['diab2'] = np.inf
        if param.nac_approx is not None:
            self.nac_approx=param.nac_approx
        else:
            self.nac_approx = None
        if self.adaptive is not None:
            self.NNnumber = int(2)
            self.modelpaths=[]
            self.modelpaths.append(param.modelpath)
            self.modelpaths.append(param.modelpaths)
            self.schnarc_init = EnsembleSchNarculator(self.dummy_crd,self.AtNames,self.modelpaths,self.device,hessian=self.hessian,nac_approx=self.nac_approx,adaptive=self.adaptive,thresholds=self.thresholds,print_uncertainty=param.print_uncertainty)
        else:
            self.NNnumber = int(1) #self.options["NNnumber"]
            self.schnarc_init = SchNarculator(self.dummy_crd,self.AtNames,param.modelpath,self.device,hessian=self.hessian,nac_approx=self.nac_approx)
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
        adaptive = float(1)
    param = "schnarc_options"
    # init SHARC_NN class 
    nn = SHARC_NN()
    print( "init SHARC_NN:  CPU time: % .3f s, wall time: %.3f s"%(time.process_time() - tc, time.time() - tt))
    # run sharc dynamics
    nn.run_sharc("input",args,adaptive, initial_param=param)
    print( "run dynamics:  CPU time: % .3f s, wall time: %.3f s"%(time.process_time() - tc, time.time() - tt))
if __name__ == "__main__":
    main()



