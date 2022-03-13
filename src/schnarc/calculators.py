from scipy import sparse 
import torch
import numpy as np
import os
import subprocess
from ase import Atoms
from ase import neighborlist
from collections import Iterable
import schnarc
from schnetpack import Properties
from schnetpack.environment import SimpleEnvironmentProvider, collect_atom_triples
import torch

class SchNarculatorError(Exception):
    pass


class SchNarculator:

    def __init__(self, positions, atom_types, modelpath,
                 param,
                 hessian=False,nac_approx=[1,None,None],adaptive=None,thresholds=1, print_uncertainty=None):

        self.device = torch.device("cuda" if param.cuda else "cpu")
        # Load model
        self.parallel = False
        self.socs_mask = param.socs_mask
        self.finish = param.finish
        self.model = self._load_model(modelpath)
        if param.socmodel is not None:
            self.socmodel = self._load_model(param.socmodel)
        else:
            self.socmodel = None
        if param.nacmodel is not None:
            self.nacmodel = self._load_model(param.nacmodel)
        else:
            self.nacmodel = None
        if param.emodel2 is not None:
            self.emodel2 = self._load_model(param.emodel2)
        else:
            self.emodel2 = None

        self.adaptive = adaptive
        self.thresholds = thresholds
        self.print_uncertainty = print_uncertainty
        if self.adaptive is not None:
            if self.parallel:
                self.n_states_dict = self.model[0].module.output_modules[0].n_states
                self.model_all = self.model
                self.model = self.model[0]
            else:
                self.n_states_dict = self.model[0].output_modules[0].n_states
                self.model_all = self.model
                self.model = self.model[0]
                #print("Parallel mode not implemented for adaptive sampling.")
        else:
            if not self.parallel:
                self.n_states_dict = self.model.output_modules[0].n_states
            else:
                self.n_states_dict = self.model.module.output_modules[0].n_states
        self.n_states = self.n_states_dict['n_states']
        self.n_singlets = self.n_states_dict['n_singlets']
        self.n_triplets = self.n_states_dict['n_triplets']
        self.n_atoms = positions.shape[0]
        self.environment_provider = param.environment_provider
        self.collect_triples = None
        if self.environment_provider == "simple":
            self.environment_provider = SimpleEnvironmentProvider()
        else:
            print("Code not implemented for different environmentproviders than simple. Adapt code for your environment provider.")
        self.threshold_dE_S=nac_approx[1]
        self.threshold_dE_T=nac_approx[2]
        self.hessian=[hessian,self.n_singlets,self.threshold_dE_S,self.n_triplets,self.threshold_dE_T,self.finish]
        self.hessian2=[False,self.n_singlets,self.threshold_dE_S,self.n_triplets,self.threshold_dE_T,self.finish]
        self.nacs_approx_method=nac_approx[0]
        if param.diss_tyro is not None:
            self.diss_tyro = True
        else:
            self.diss_tyro = False
        # Enable the hessian flag if requested and set need_hessian in old models
        if self.parallel:
            #soc model is assumed to be not trained parallel #TODO make general
            if not hasattr(self.model.module.output_modules, 'need_hessian'):
                self.model.module.output_modules.need_hessian = self.hessian
                if self.emodel2 is not None:
                   self.emodel2.module.output_modules.need_hessian=[False,self.n_singlets,0,self.n_triplets,0,self.finish]
                if self.socmodel is not None:
                   self.socmodel.module.output_modules.need_hessian=[False,self.n_singlets,0,self.n_triplets,0,self.finish]
                if self.nacmodel is not None:
                   self.nacmodel.module.output_modules.need_hessian=[False,self.n_singlets,0,self.n_triplets,0,self.finish]
            if not hesattr(self.model.module.output_modules,'order'):
                self.model.module.output_modules.order = False
                if self.emodel2 is not None:
                    self.emodel2.module.output_modules.order = False
 
            if hessian:
                if schnarc.data.Properties.energy in self.model.module.output_modules[0].output_dict:
                    self.model.module.output_modules[0].output_dict[schnarc.data.Properties.energy].return_hessian = self.hessian
                    if self.emodel2 is not None:
                       self.emodel2.module.output_modules[0].output_dict[schnarc.data.Properties.energy].return_hessian=[False,self.n_singlets,0,self.n_triplets,0,self.finish]
                    if self.socmodel is not None:
                       self.socmodel.module.output_modules[0].output_dict[schnarc.data.Properties.socs].return_hessian=[False,self.n_singlets,0,self.n_triplets,0,self.finish]
                    if self.nacmodel is not None:
                       self.nacmodel.module.output_modules[0].output_dict[schnarc.data.Properties.socs].return_hessian=[False,self.n_singlets,0,self.n_triplets,0,self.finish]
        else:
            if not hasattr(self.model.output_modules, 'need_hessian'):
                self.model.output_modules.need_hessian = self.hessian
                if self.emodel2 is not None:
                   self.emodel2.output_modules.need_hessian=self.hessian2 #[False,self.n_singlets,0,self.n_triplets,0]
                if self.socmodel is not None:
                   self.socmodel.output_modules.need_hessian=self.hessian2 #[False,self.n_singlets,0,self.n_triplets,0]

                if self.nacmodel is not None:
                   self.nacmodel.output_modules.need_hessian=self.hessian2 #[False,self.n_singlets,0,self.n_triplets,0]
            if not hasattr(self.model.output_modules, 'order'):
                self.model.output_modules.order  = False
                if self.emodel2 is not None:
                    self.model.output_modules.order = False
            if hessian:
                if schnarc.data.Properties.energy in self.model.output_modules[0].output_dict:
                    self.model.output_modules[0].output_dict[schnarc.data.Properties.energy].return_hessian = self.hessian
                    if self.emodel2 is not None:
                       self.emodel2.output_modules[0].output_dict[schnarc.data.Properties.energy].return_hessian= self.hessian2 #[False,self.n_singlets,0,self.n_triplets,0]
                    if self.socmodel is not None:
                       self.socmodel.output_modules[0].output_dict[schnarc.data.Properties.socs].return_hessian=self.hessian2 #[False,self.n_singlets,0,self.n_triplets,0]
                    if self.nacmodel is not None:
                       self.nacmodel.output_modules[0].output_dict[schnarc.data.Properties.nacs].return_hessian=self.hessian2 #[False,self.n_singlets,0,self.n_triplets,0]

        self.molecule = Atoms(atom_types, positions)

    def calculate(self, sharc_outputs):
        # Format inputs
        schnet_inputs = self._sharc2schnet(sharc_outputs)
        # Perform prediction
        schnet_outputs = self._calculate(schnet_inputs)
        #schnet_inputs["energy"] = schnet_outputs["energy"]
        #schnet_inputs["nac_energy"] = schnet_outputs["energy"]
        if self.socmodel is not None:
            schnet_outputs["_socs"]=True
        else:
            schnet_outputs["_socs"]  = False
        if self.nacmodel is not None:
            schnet_outputs["_nacs"]=True
        else:
            schnet_outputs["_nacs"] = False
        schnet_socoutputs,schnet_nacoutputs = self._calculate2(schnet_inputs,schnet_outputs["_socs"],schnet_outputs["_nacs"])
        # Format outputs
        if self.emodel2 is not None:
            schnet_outputs2 = self._calculate3(schnet_inputs)
            sharc_inputs = self._schnet2sharc(schnet_outputs,schnet_socoutputs,schnet_nacoutputs,schnet_outputs2,self.thresholds)

        else:
            sharc_inputs = self._schnet2sharc(schnet_outputs,schnet_socoutputs,schnet_nacoutputs,None,self.thresholds)
        """else:
            if self.emodel2 is not None:
                schnet_outputs2 = self._calculate3(schnet_inputs)
                sharc_inputs = self._schnet2sharc(schnet_outputs,None,None,schnet_outputs2,self.thresholds)
                if self.adaptive is not None:
                    sharc_inputs2 = self._schnet2sharc(schnet_outputs2,None,None,schnet_outputs,self.thresholds)
                    for key in sharc_inputs.keys():
                        if key in sharc_inputs2.keys():
                            sharc_inputs[key] = ( ( np.array(sharc_inputs[key]) + np.array(sharc_inputs2[key])) / 2).tolist()
            else:
                sharc_inputs = self._schnet2sharc(schnet_outputs,None,None,self.thresholds)"""
        return sharc_inputs

    def _load_model(self, modelpath):
        if os.path.isdir(modelpath):
            modelpath = os.path.join(modelpath, 'best_model')

        if not torch.cuda.is_available():
            model = torch.load(modelpath, map_location='cpu')
        else:
            model = torch.load(modelpath)
        # Check if parallel
        self.parallel = isinstance(model, torch.nn.DataParallel)
        model = model.to(self.device)
        return model


    def eval_dis(self,atoms,natoms):
        atype=atoms.get_atomic_numbers()
        # computes distances of all H atoms
        # takes the closest atom and if the distance is larger
        # than a threshold == set to 2 A = 3.78 Bohr
        # only for tyrosine with the artificial data points
        d=3.78
        for i in range(natoms):
            #cutoff=(list((np.arange(natoms)+1)*np.inf))
            #nl = neighborlist.NeighborList(cutoff)
            #nl.update(atoms)
            #a,b=nl.get_neighbors(1)
            if atype[i]==1:
                 dist=atoms.get_distances(i,np.arange(natoms))
                 #get the smallest distance
                 distance=np.sort(dist)[1]
                 dist_atom_index= [i for i,j  in enumerate(dist) if j==distance]
                 if distance > 3.78:
                     print("H Atom might dissociate")
                     #set distance and fix non-hydrogen-type atom
                     atoms.set_distance(i,dist_atom_index[0],d,fix=1)
                     new_positions=atoms.get_positions()
                 else:
                     new_positions=atoms.get_positions()
            #matrix = nl.get_connectivity_matrix()
        return new_positions

    def _sharc2schnet(self, sharc_output):
        # Update internal structure with new Shark positions
        self.molecule.positions = np.array(sharc_output)

        schnet_inputs = dict()
        # Elemental composition
        schnet_inputs[Properties.Z] = torch.LongTensor(self.molecule.numbers.astype(np.int))
        schnet_inputs[Properties.atom_mask] = torch.ones_like(schnet_inputs[Properties.Z]).float()
        # Set positions
        original_positions = self.molecule.positions.astype(np.float32)
        if self.diss_tyro == True:
            atoms = Atoms(self.molecule.numbers.astype(np.int),self.molecule.positions)
            natoms=len(self.molecule.numbers.astype(np.int))
            original_positions=self.eval_dis(atoms,natoms)
        schnet_inputs[Properties.R] = torch.FloatTensor(original_positions)

        # get atom environment
        nbh_idx, offsets = self.environment_provider.get_environment(self.molecule)
        # Get neighbors and neighbor mask
        mask = torch.FloatTensor(nbh_idx) >= 0
        schnet_inputs[Properties.neighbor_mask] = mask.float()
        schnet_inputs[Properties.neighbors] = torch.LongTensor(nbh_idx.astype(np.int)) * mask.long()
        # Get cells
        schnet_inputs[Properties.cell] = torch.FloatTensor(self.molecule.cell.astype(np.float32))
        schnet_inputs[Properties.cell_offset] = torch.FloatTensor(offsets.astype(np.float32))
        # If requested get masks and neighbor lists for neighbor pairs
        if self.collect_triples is not None:
            nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(nbh_idx)
            schnet_inputs[Properties.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
            schnet_inputs[Properties.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))
            schnet_inputs[Properties.neighbor_pairs_mask] = torch.ones_like(
                schnet_inputs[Properties.neighbor_pairs_j]).float()
        # Add batch dimension and move to CPU/GPU
        for key, value in schnet_inputs.items():
            schnet_inputs[key] = value.unsqueeze(0).to(self.device)

        return schnet_inputs

    def _calculate2(self, schnet_inputs,_socs,_nacs):
        if self.parallel:
            if _socs==True:
                schnet_socoutputs = self.socmodel.module(schnet_inputs)
            if _nacs == True:
                schnet_nacoutputs = self.nacmodel.module(schnet_inputs)
        else:
            if _socs==True:
                schnet_socoutputs = self.socmodel(schnet_inputs)
            if _nacs == True:
                schnet_nacoutputs = self.nacmodel(schnet_inputs)
        # Move to cpu, detach from computational graph and convert to numpy
        if _socs==True:
            schnet_socoutputs["socs"]=schnet_socoutputs["socs"].cpu().detach().numpy()
            schnet_socoutputs["socs"][0]=schnet_socoutputs["socs"][0]*self.socs_mask
        else:
            schnet_socoutputs = None
        if _nacs == True:
            schnet_nacoutputs["nacs"]=schnet_nacoutputs["nacs"].cpu().detach().numpy()
        else:
            schnet_nacoutputs = None
        return schnet_socoutputs, schnet_nacoutputs

    def _calculate3(self, schnet_inputs):
        if self.parallel:
            schnet_outputs = self.emodel2.module(schnet_inputs)
        else:
            schnet_outputs = self.emodel2(schnet_inputs)
        # Move to cpu, detach from computational graph and convert to numpy
        for key, value in schnet_outputs.items():
            schnet_outputs[key] = value.cpu().detach().numpy()
        return schnet_outputs

    def _calculate(self, schnet_inputs):
        if self.parallel:
            schnet_outputs = self.model.module(schnet_inputs)
        else:
            schnet_outputs = self.model(schnet_inputs)
        # Move to cpu, detach from computational graph and convert to numpy
        for key, value in schnet_outputs.items():
            schnet_outputs[key] = value.cpu().detach().numpy()
        return schnet_outputs

    def _schnet2sharc(self, schnet_outputs, schnet_socoutputs, schnet_nacoutputs, schnet_outputs2, thresholds):
        #creates dictionary for pySHARC
        QMout={}
        if schnet_outputs2 is not None:
            MAE = np.mean(np.abs(schnet_outputs["energy"] - schnet_outputs2["energy"]),axis=0)
            print("MAE model S0:S5, T1:T8:",MAE)
            if np.mean(MAE) >= thresholds["energy"]:
                print("Terminate model at current step due to large error between models: %f" %np.mean(MAE))
                exit()

        #iterate over all properties
        hamiltonian_update = False
        QMout['dm'] = [[[0.0 for k in range(int(self.n_singlets+3*self.n_triplets))] for j in range(int(self.n_singlets+3*self.n_triplets))] for i in range(3)]
        QMout['nacdr']= [[[[0.0 for xyz in range(3)] for iatom in range(self.n_atoms)] for istate in range(int(self.n_singlets+3*self.n_triplets))] for jstate in range(int(self.n_singlets+3*self.n_triplets))]
        #sort
        if self.n_triplets==int(0):
            index=np.argsort(schnet_outputs['energy'][0])
        else:
            index=np.argsort(schnet_outputs['energy'][0][0:self.n_singlets])
            indext=np.argsort(schnet_outputs['energy'][0][self.n_singlets:int(self.n_singlets+self.n_triplets*3)])+self.n_singlets

        for i,prop in enumerate(schnet_outputs):
            if prop == "energy":
                hamiltonian = np.zeros((self.n_singlets+self.n_triplets*3,self.n_singlets+self.n_triplets*3),dtype=complex)
                for istate in range(self.n_singlets):
                    hamiltonian[istate][istate] = complex(schnet_outputs['energy'][0][index[istate]], 0.000)
                for istate in range(self.n_singlets,self.n_singlets+self.n_triplets):
                    hamiltonian[istate][istate] = complex(schnet_outputs['energy'][0][indext[istate-self.n_singlets]], 0.000)
                    hamiltonian[istate+self.n_triplets][istate+self.n_triplets] = complex(schnet_outputs['energy'][0][indext[istate-self.n_singlets]], 0.000)
                    hamiltonian[istate+self.n_triplets*2][istate+self.n_triplets*2] = complex(schnet_outputs['energy'][0][indext[istate-self.n_singlets]], 0.000)
                hamiltonian_list = np.array(hamiltonian).tolist()
                QMout['h'] = hamiltonian_list
            elif prop == "forces":
                n_atoms = self.n_atoms
                gradients = np.zeros( (self.n_singlets+self.n_triplets*3, n_atoms, 3) )
                for istate in range(self.n_singlets):
                    gradients[istate] = -schnet_outputs['forces'][0][index[istate]]
                for istate in range(self.n_singlets,self.n_singlets+self.n_triplets):
                    gradients[istate] = -schnet_outputs['forces'][0][indext[istate-self.n_singlets]]
                    gradients[istate+self.n_triplets] = -schnet_outputs['forces'][0][indext[istate-self.n_singlets]]
                    gradients[istate+self.n_triplets*2] = -schnet_outputs['forces'][0][indext[istate-self.n_singlets]]
                QMout['grad'] = np.array(gradients).tolist()
            elif prop == "dipoles":
                dipole_matrix = [[[0.0 for k in range(self.n_singlets+3*self.n_triplets)] for j in range(self.n_singlets+3*self.n_triplets)] for i in range(3)]
                for xyz in range(3):
                    iterator=-1
                    for istate in range(self.n_singlets):
                        for jstate in range(istate,self.n_singlets):
                            iterator+=1
                            dipole_matrix[xyz][index[istate]][index[jstate]] = schnet_outputs['dipoles'][0][iterator][xyz]
                            dipole_matrix[xyz][index[jstate]][index[istate]] = schnet_outputs['dipoles'][0][iterator][xyz]
                    for istate in range(self.n_singlets,self.n_triplets):
                        for jstate in range(istate,self.n_singlets+self.n_triplets):
                            iterator+=1
                            dipole_matrix[xyz][index[istate]][index[jstate]] = schnet_outputs['dipoles'][0][iterator][xyz]
                            dipole_matrix[xyz][index[jstate]][index[istate]] = schnet_outputs['dipoles'][0][iterator][xyz]
                            dipole_matrix[xyz][index[istate+self.n_triplets]][index[jstate+self.n_triplets]] = schnet_outputs['dipoles'][0][iterator][xyz]
                            dipole_matrix[xyz][index[jstate+self.n_triplets]][index[istate+self.n_triplets]] = schnet_outputs['dipoles'][0][iterator][xyz]
                            dipole_matrix[xyz][index[istate+2*self.n_triplets]][index[jstate+2*self.n_triplets]] = schnet_outputs['dipoles'][0][iterator][xyz]
                            dipole_matrix[xyz][index[jstate+2*self.n_triplets]][index[istate+2*self.n_triplets]] = schnet_outputs['dipoles'][0][iterator][xyz]
                dipole_list = np.array(dipole_matrix).tolist()
                QMout.update( { 'dm' : dipole_list } )
            elif ( schnet_socoutputs is not None and prop == "_socs") or prop=="old_socs" or prop=="socs":
                hamiltonian_update = True

            elif prop == "nacs" or ( prop == "_nacs" and schnet_nacoutputs is not None ):# and prop == "_nacs":
                if schnet_nacoutputs is not None:
                    schnet_outputs["nacs"] = schnet_nacoutputs["nacs"] 
                    it=-1
                    for istate in range(self.n_singlets):
                        for jstate in range(istate+1,self.n_singlets):
                            it+=1
                            schnet_outputs["nacs"][it] = schnet_outputs["nacs"][0,it] / abs(schnet_outputs["energy"][0,istate] - schnet_outputs["energy"][0,jstate])
                    for istate in range(self.n_triplets):
                        for jstate in range(self.n_triplets):
                            it+=1
                            schnet_outputs["nacs"][it] = schnet_outputs["nacs"][it] / abs(schnet_outputs["energy"][istate] - schnet_outputs["energy"][jstate])
                nonadiabatic_couplings = np.zeros((self.n_singlets+3*self.n_triplets,self.n_singlets+3*self.n_triplets,self.n_atoms,3))
                iterator = -1
                for istate in range(self.n_singlets):
                    for jstate in range(istate+1,self.n_singlets):
                        iterator += 1
                        if istate==int(1) and jstate==int(2):
                            nonadiabatic_couplings[index[istate]][index[jstate]] = schnet_outputs['nacs'][0][iterator]
                            nonadiabatic_couplings[index[jstate]][index[istate]] = -schnet_outputs['nacs'][0][iterator]
                        else:
                            nonadiabatic_couplings[index[istate]][index[jstate]] = schnet_outputs['nacs'][0][iterator]
                            nonadiabatic_couplings[index[jstate]][index[istate]] = -schnet_outputs['nacs'][0][iterator]
                            #nonadiabatic_couplings[istate][jstate][:][:] = 0
                            #nonadiabatic_couplings[jstate][istate][:][:] = 0

                for istate in range(self.n_singlets, self.n_singlets+self.n_triplets):
                    for jstate in range(istate+1, self.n_triplets+self.n_singlets):
                        iterator += 1
                        for itriplet in range(3):
                          if istate==int(3) and jstate==int(5):
                              nonadiabatic_couplings[istate+self.n_triplets*itriplet][jstate+self.n_triplets*itriplet] = schnet_outputs['nacs'][0][iterator]
                              nonadiabatic_couplings[jstate+self.n_triplets*itriplet][istate+self.n_triplets*itriplet] = -schnet_outputs['nacs'][0][iterator]
                          else:
                              nonadiabatic_couplings[istate+self.n_triplets*itriplet][jstate+self.n_triplets*itriplet][:][:] = schnet_outputs['nacs'][0][iterator][:][:] #SO2 0
                              nonadiabatic_couplings[jstate+self.n_triplets*itriplet][istate+self.n_triplets*itriplet][:][:] = -schnet_outputs['nacs'][0][iterator][:][:] # SO2 0
                nacdr = np.array(nonadiabatic_couplings).tolist()

                QMout.update( { 'nacdr' : nacdr } )
        if hamiltonian_update == True:
            iterator = -1
            hamiltonian_soc = np.zeros((self.n_singlets+3*self.n_triplets,self.n_singlets+3*self.n_triplets),dtype=complex)
            for istate in range(self.n_singlets+3*self.n_triplets):
                for jstate in range(istate+1,self.n_singlets+3*self.n_triplets):
                    iterator+=1
                    if schnet_socoutputs is not None:
                        hamiltonian_soc[istate][jstate] = complex(schnet_socoutputs['socs'][0][iterator*2],schnet_socoutputs['socs'][0][iterator*2+1])
                    else:
                        hamiltonian_soc[istate][jstate] = complex(schnet_outputs['socs'][0][iterator*2],schnet_outputs['socs'][0][iterator*2+1])
            for istate in range(self.n_singlets):
                for jstate in range(istate+1,self.n_singlets):
                    hamiltonian_soc[istate][jstate] = complex(0.000,0.000)
            hamiltonian_soc = hamiltonian_soc + hamiltonian_soc.T
            hamiltonian_full = hamiltonian + hamiltonian_soc
            hamiltonian_list = np.array(hamiltonian_full).tolist()
            QMout.update( { 'h' : hamiltonian_list } )

        #get nac-vector as approximation from hessian and energy gap
        if self.hessian[0] == True:
            #delta Hessian
            #get magnitude by scaling of hopping direction with energy gap
            dH_2=[]
            all_magnitude=[]
            indexh = -1
            eigenvalue_hopping_direction = np.zeros( (self.n_singlets + self.n_triplets, self.n_singlets + self.n_triplets,1) )
            nacs_approx = np.zeros( (self.n_singlets+3*self.n_triplets, self.n_singlets+3*self.n_triplets, self.n_atoms, 3) )
            hopping_direction = np.zeros( (self.n_singlets + self.n_triplets, self.n_singlets + self.n_triplets, self.n_atoms, 3) )
            for istate in range(self.n_singlets):
                for jstate in range(istate+1,self.n_singlets):
                  if np.abs(np.real(hamiltonian[index[istate]][index[istate]])-np.real(hamiltonian[index[jstate]][index[jstate]])) <= self.threshold_dE_S:
                    indexh+=1
                    Hi=schnet_outputs['hessian'][0][index[istate]]
                    dE=(schnet_outputs['energy'][0][index[istate]]-schnet_outputs['energy'][0][index[jstate]])
                    if dE == 0:
                      dE=0.0000000001
                    Hj=schnet_outputs['hessian'][0][index[jstate]]
                    GiGi=np.dot(-schnet_outputs['forces'][0][index[istate]].reshape(-1,1),-schnet_outputs['forces'][0][index[istate]].reshape(-1,1).T)
                    GjGj=np.dot(-schnet_outputs['forces'][0][index[jstate]].reshape(-1,1),-schnet_outputs['forces'][0][index[jstate]].reshape(-1,1).T)
                    GiGj=np.dot(-schnet_outputs['forces'][0][index[istate]].reshape(-1,1),-schnet_outputs['forces'][0][index[jstate]].reshape(-1,1).T)
                    GjGi=np.dot(-schnet_outputs['forces'][0][index[jstate]].reshape(-1,1),-schnet_outputs['forces'][0][index[istate]].reshape(-1,1).T)

                    G_diff = 0.5*(-schnet_outputs['forces'][0][index[istate]]+schnet_outputs['forces'][0][index[jstate]])
                    G_diff2= np.dot(G_diff.reshape(-1,1),G_diff.reshape(-1,1).T)

                    dH_2_ij = 0.5*(dE*(Hi-Hj) + GiGi + GjGj - 2*GiGj)
                    dH_2.append(dH_2_ij)
                    magnitude = dH_2_ij/2-G_diff2
                    #magnitude_ZN=Hi-Hj
                    all_magnitude.append(magnitude)
                    #SVD
                    #u,s,vh = np.linalg.svd(deltaHessian_2[index])
                    u,s,vh=np.linalg.svd(all_magnitude[indexh])
                    ev=vh[0]
                    #get one phase
                    e=max(ev[0:2].min(),ev[0:2].max(),key=abs)
                    if e>=0.0:
                        pass
                    else:
                        ev=-ev
                    ew=s[0]
                    iterator = -1
                    for iatom in range(self.n_atoms):
                        for xyz in range(3):
                            iterator+=1
                            hopping_direction[istate][jstate][iatom][xyz] = ev[iterator]
                            hopping_direction[jstate][istate][iatom][xyz] = -ev[iterator]
                    #for SO2
                    if self.nacs_approx_method == int(2):
                        if istate==int(0) and jstate==int(1):
                          hopping_magnitude =np.sqrt(ew)/dE
                        else:
                          hopping_magnitude=0.0
                    else:
                        hopping_magnitude =np.sqrt(ew)/dE

                    nacs_approx[istate][jstate][:][:] = hopping_direction[istate][jstate] * hopping_magnitude
                    nacs_approx[jstate][istate] = - nacs_approx[istate][jstate]
            for istate in range(self.n_singlets,self.n_singlets+self.n_triplets):
                for jstate in range(istate+1, self.n_singlets+self.n_triplets):
                  if np.abs(np.real(hamiltonian[indext[istate-self.n_singlets]][indext[istate-self.n_singlets]])-np.real(hamiltonian[indext[jstate-self.n_singlets]][indext[jstate-self.n_singlets]])) <= self.threshold_dE_T:
                    Hi=schnet_outputs['hessian'][0][indext[istate-self.n_singlets]]
                    dE=np.abs(schnet_outputs['energy'][0][indext[istate-self.n_singlets]]-schnet_outputs['energy'][0][indext[jstate-self.n_singlets]])
                    Hj=schnet_outputs['hessian'][0][indext[jstate-self.n_singlets]]
                    if dE == 0:
                      dE=0.0000000001
                    GiGi=np.dot(-schnet_outputs['forces'][0][indext[istate-self.n_singlets]].reshape(-1,1),-schnet_outputs['forces'][0][indext[istate-self.n_singlets]].reshape(-1,1).T)
                    GjGj=np.dot(-schnet_outputs['forces'][0][indext[jstate-self.n_singlets]].reshape(-1,1),-schnet_outputs['forces'][0][indext[jstate-self.n_singlets]].reshape(-1,1).T)
                    GiGj=np.dot(-schnet_outputs['forces'][0][indext[istate-self.n_singlets]].reshape(-1,1),-schnet_outputs['forces'][0][indext[jstate-self.n_singlets]].reshape(-1,1).T)
                    GjGi=np.dot(-schnet_outputs['forces'][0][indext[jstate-self.n_singlets]].reshape(-1,1),-schnet_outputs['forces'][0][indext[istate-self.n_singlets]].reshape(-1,1).T)

                    G_diff = 0.5*(-schnet_outputs['forces'][0][indext[istate-self.n_singlets]]+schnet_outputs['forces'][0][indext[jstate-self.n_singlets]])
                    G_diff2= np.dot(G_diff.reshape(-1,1),G_diff.reshape(-1,1).T)

                    dH_2_ij = 0.5*(dE*(Hi-Hj) + GiGi + GjGj - 2*GiGj)
                    dH_2.append(dH_2_ij)
                    magnitude = dH_2_ij/2-G_diff2
                    all_magnitude.append(magnitude)
                    #SVD
                    indexh+=1
                    u,s,vh = np.linalg.svd(all_magnitude[indexh])
                    ev=vh[0]
                    #get one phase
                    e=max(ev[0:2].min(),ev[0:2].max(),key=abs)
                    if e>=0.0:
                        pass
                    else:
                        ev=-ev
                    ew=s[0]
                    iterator = -1
                    for iatom in range(self.n_atoms):
                        for xyz in range(3):
                            iterator += 1
                            hopping_direction[istate][jstate][iatom][xyz] = ev[iterator]
                            hopping_direction[jstate][istate][iatom][xyz] = -ev[iterator]
                    #for SO2
                    if self.nacs_approx_method == int(2):
                        if istate==int(3) and jstate==int(5):
                          hopping_magnitude=np.sqrt(ew)/dE
                        else:
                          hopping_magnitude=0.0
                    else:
                        hopping_magnitude =np.sqrt(ew)/dE
                    for itriplet in range(3):
                        nacs_approx[istate+self.n_triplets*itriplet][jstate+self.n_triplets*itriplet] = (hopping_direction[istate][jstate]) * hopping_magnitude
                        nacs_approx[jstate+self.n_triplets*itriplet][istate+self.n_triplets*itriplet] = - nacs_approx[istate+self.n_triplets*itriplet][jstate+self.n_triplets*itriplet]

            #NOTE NAC approx 2 is only for SO2 involving 3 singlets and 3 triplets!
            deltaHessian_2=np.array(dH_2)
            all_magnitude=np.array(all_magnitude)
            #get magnitude by scaling of hopping direction with energy gap
            nacs_approx = nacs_approx.tolist()
            QMout.update( { 'nacdr' : nacs_approx } )

        return QMout


class EnsembleSchNarculator(SchNarculator):

    def __init__(self, positions, atom_types, modelpaths, param,
                 #device=torch.device('cpu'),
                 #environment_provider=SimpleEnvironmentProvider(),
                 #collect_triples=False,
                 hessian=False,nac_approx=[1,None,None],adaptive=None,thresholds=1,print_uncertainty=None):
        # Check whether a list of modelpaths has been passed
        if not isinstance(modelpaths, Iterable):
            raise SchNarculatorError('List of modelpaths required for ensemble calculator.')

        super(EnsembleSchNarculator, self).__init__(positions, atom_types, modelpaths, param,
        #device, environment_provider,collect_triples,
        hessian=hessian,nac_approx=nac_approx,adaptive=adaptive,thresholds=thresholds,print_uncertainty=print_uncertainty)
                                                    #device=device,
                                                    #environment_provider=environment_provider,
                                                    #collect_triples=collect_triples, hessian=hessian, nac_approx=nac_approx)

        self.n_models = len(self.model_all)
        self.uncertainty = {}
    def _load_model(self, modelpath):
        self.models = []
        for path in modelpath:
            if os.path.isdir(path):
                path = os.path.join(path, 'best_model')
            if not torch.cuda.is_available():
                model = torch.load(path, map_location='cpu')
            else:
                model = torch.load(path)
            self.parallel = isinstance(model, torch.nn.DataParallel)
            self.models.append(model.to(self.device))
        return self.models

    def _calculate(self, schnet_inputs):
        ensemble_results = {}
        for model in self.models:
            if self.parallel == True:
                schnet_outputs = model.module(schnet_inputs)
            else:
                schnet_outputs = model(schnet_inputs)
            for prop in schnet_outputs:
                if prop in ensemble_results:
                    ensemble_results[prop].append(schnet_outputs[prop].cpu().detach().numpy())
                else:
                    ensemble_results[prop] = []
                    ensemble_results[prop].append(schnet_outputs[prop].cpu().detach().numpy())

        results = {}

        for prop in ensemble_results:
            if prop == "hessian" or prop == "diab" or prop == "diab2":
                ensemble_results[prop] = np.array(ensemble_results[prop])
                results[prop] =np.mean(ensemble_results[prop],axis=0)
            else:
                ensemble_results[prop] = np.array(ensemble_results[prop])
                results[prop] =np.mean(ensemble_results[prop],axis=0)
                self.uncertainty[prop] = np.mean(np.std(ensemble_results[prop],axis=0))
                if self.print_uncertainty==True:
                    print(self.uncertainty[prop], "property:", prop)
                if self.uncertainty[prop] >= self.thresholds[prop]:
                    print("Terminate trajectory due to large error between models")
                    exit()

        return results


class Queuer:
    QUEUE_FILE = """
#!/usr/bin/env bash
##############################
#$ -cwd
#$ -V
#$ -q {queue}
#$ -N {jobname}
#$ -t 1-{array_range}
#$ -tc {concurrent}
#$ -S /bin/bash
#$ -e /dev/null
#$ -o /dev/null
#$ -r n
#$ -sync y
##############################

# Adapt here
"""

    def __init__(self, queue, executable, concurrent=100, basename='input', cleanup=True):
        self.queue = queue
        self.executable = executable
        self.concurrent = concurrent
        self.basename = basename
        self.cleanup = cleanup

    def submit(self, input_files, current_compdir):
        jobname = os.path.basename(current_compdir)
        compdir = os.path.abspath(current_compdir)
        n_inputs = len(input_files)

        submission_command = self._create_submission_command(n_inputs, compdir, jobname)

        script_name = os.path.join(current_compdir, 'submit.sh')
        with open(script_name, 'w') as submission_script:
            submission_script.write(submission_command)

        computation = subprocess.Popen(['qsub', script_name], stdout=subprocess.PIPE)
        computation.wait()

        if self.cleanup:
            os.remove(script_name)

    def _create_submission_command(self, n_inputs, compdir, jobname):
        raise NotImplementedError
