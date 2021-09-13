import schnetpack as spk
import torch
import torch.nn as nn

from torch.autograd import grad
from schnetpack import Properties
from schnetpack.atomistic import Atomwise

import schnarc
from schnarc.data import Properties


class StateModel(nn.Module):
    def __init__(self, representation, output_modules, mapping=None):
        super(StateModel, self).__init__()

        self.representation = representation

        self._get_mapping(output_modules, mapping)

        if not isinstance(output_modules, dict):
            raise ValueError('Dictionary expected for output_modules')

        self.output_modules = nn.ModuleDict(output_modules)

        # Set derivative flag
        
        self.requires_dr = False
        for o in self.output_modules:
            if self.output_modules[o].requires_dr:
                self.requires_dr = True
                break

        self.module_results = {}

    def forward(self, inputs):

        if self.requires_dr:
            inputs[spk.Properties.R].requires_grad_()

        inputs['representation'] = self.representation(inputs)

        module_results = {}
        # TODO: double for might not be the nicest. Worst case revert to previous dict of dict
        #   This version has advantage, that everthing is accessible in output and no parallel model collisions
        #   will happen...
        for output_module in self.output_modules:
            results = self.output_modules[output_module](inputs)
            for entry in results:
                tag = f'{output_module}:{entry}'
                module_results[tag] = results[entry]

        module_results = self._map_outputs(module_results)

        return module_results

    def _get_mapping(self, output_modules, mapping):
        # Generate a default mapping
        self.mapping = {p: f'{p}:y' for p in output_modules.keys()}

        # Update default mapping with custom instructions
        if mapping is not None:
            for p in mapping.keys():
                self.mapping[p] = f'{mapping[p][0]}:{mapping[p][1]}'

    def _map_outputs(self, module_results):

        for entry in self.mapping:
            module_results[entry] = module_results.pop(self.mapping[entry])
        return module_results


class MultiStatePropertyModel(nn.Module):
    """
    Basic hardcoded model for different properties. This is mainly intended for production and should be updäted with
    those modules yieling the best results for all properties.

    mean, stddev and atomrefs only for energies
    """
    def __init__(self, n_in, n_states, properties,n_neurons=None, mean=None, stddev=None, atomref=None, need_atomic=False, n_layers=2,
                 inverse_energy=0, real=False ):

        super(MultiStatePropertyModel, self).__init__()

        self.n_in = n_in
        self.n_states = n_states
        self.n_singlets = n_states['n_singlets']
        self.n_triplets = n_states['n_triplets']
        self.nmstates = n_states['n_states']
        self.properties = properties
        self.need_atomic = need_atomic
        self.real = real
        self._init_property_flags(properties)
        self.order = n_states["order"]
        # Flag for computation of nacs
        self.inverse_energy = inverse_energy
        self.finish = n_states["finish"]
        if self.need_forces == True:
            create_graph=True
            self.derivative = True
        else:
            create_graph=False
            self.derivative = None
        # Construct default mean and stddevs if passed None
        if mean is None or stddev is None:
            mean = {p: None for p in properties}
            stddev = {p: None for p in properties}

        # Disable stress
        self.stress = False

        outputs = {}
        # Energies and forces
        if self.need_energy or self.need_forces:
            try:
                atomref = atomref[Properties.energy]
            except:
                atomref = None
            energy_module = MultiEnergy(n_in, self.n_singlets + self.n_triplets, aggregation_mode='sum',
                                        return_force=self.need_forces,
                                        n_neurons=n_neurons,
                                        return_contributions=self.need_atomic, mean=mean[Properties.energy],
                                        stddev=stddev[Properties.energy],
                                        atomref=atomref, create_graph=create_graph, n_layers=n_layers,order=self.order,n_triplets=self.n_triplets)

            #print(n_neurons)
            outputs[Properties.energy] = energy_module

        # Dipole moments and transition dipole moments
        if self.need_dipole:
            n_dipoles = int((self.n_singlets+self.n_triplets) * (self.n_singlets+self.n_triplets + 1) / 2)  # Between ALL states
            dipole_module = MultiDipole(n_in, n_states, n_layers=n_layers, n_neurons=n_neurons)
            outputs[Properties.dipole_moment] = dipole_module

        # Nonadiabatic couplings
        if self.need_nacs:
            self.derivative = True 
            #n_couplings = int(self.n_singlets * (self.n_singlets - 1) / 2 + self.n_triplets * (self.n_triplets - 1) / 2)  # Between all different states
            nacs_module = MultiNac(n_in, n_states, n_layers=n_layers, use_inverse=self.inverse_energy, n_neurons=n_neurons)
            outputs[Properties.nacs] = nacs_module

        # Spinorbit couplings
        if self.need_socs:
            if "n_socs" in n_states:
                n_socs = int(n_states["n_socs"])
            else:
                n_socs = int(
                (self.n_singlets+3*self.n_triplets) * (self.n_singlets+3*self.n_triplets-1))  # Between all different states - including imaginary numbers
            socs_module = MultiSoc(n_in, n_socs, n_layers=n_layers, real=real, mean=None,
                                   stddev=None, n_neurons=n_neurons,create_graph=False)
            outputs[Properties.old_socs] = socs_module
            outputs[Properties.socs] = socs_module

        self.output_dict = nn.ModuleDict(outputs)

    def _init_property_flags(self, properties):
        self.need_energy = Properties.energy in properties
        self.need_forces = Properties.forces in properties
        self.need_dipole = Properties.dipole_moment in properties
        self.need_nacs = Properties.nacs in properties
        self.need_socs = Properties.socs in properties
        if self.need_socs:
            pass
        else:
            self.need_socs = Properties.old_socs in properties

    def forward(self, inputs):

        outputs = {}
        for prop, model in self.output_dict.items():
            result = model(inputs)
            outputs[prop] = result['y']
            if "energy" in inputs and self.inverse_energy == 1:
                #for FULVENE 
                #inputs['nac_energy'] = result['y'].detach()
                # Use reference energy during training
                inputs['nac_energy'] = inputs['energy']
                # And predicted during production
            elif prop == "energy" and self.inverse_energy == 2: # or self.inverse_energy == True:
                inputs['nac_energy'] = result['y'].detach()
            elif self.inverse_energy == 0:
                pass
            elif self.inverse_energy == 1 and ("energy" not in inputs and prop != "energy"):
                pass
            elif self.inverse_energy==1 and "energy" not in inputs and prop == "energy":
                inputs['nac_energy']= result['y'].detach()
                #print("You cannot use this method without reference energies in the Training mode.")
            else:
                inputs['nac_energy'] = result['y'].detach()

            if prop == Properties.energy and self.need_forces:
                outputs[Properties.forces] = result['dydx']
            if prop == Properties.energy and 'd2ydx2' in result:
                outputs[Properties.hessian] = result['d2ydx2']

            if prop == Properties.nacs:
                outputs[prop] = result['dydx']
                #outputs["oldnacs"] = result["dydx_old"]
                #outputs['diab'] = result['y']
                #outputs['diab2'] = result['y2']
        return outputs


class MultiStateError(Exception):
    pass




class MultiState(Atomwise):

    def __init__(self, n_in, n_states, aggregation_mode='sum', n_layers=2, n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus, return_contributions=False, create_graph=False,
                 return_force=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None,
                 standardize_after=False, return_hessian=[False,1,0.018,1,0.036,False], order=None,n_triplets = 0):

        self.n_states = n_states
        self.return_hessian = return_hessian
        self.order = order
        self.n_triplets = n_triplets
        self.finish = self.return_hessian[5]
        if self.return_hessian[0] and not return_force:
            raise MultiStateError('Computation of forces required for Hessian.')

            # Don't standardize in the model, but provide access to the initialized module
        if standardize_after and mean is not None and stddev is not None:
            self.standardize = spk.nn.ScaleShift(mean, stddev)
            curr_mean = None
            curr_stddev = None
        else:
            self.standardize = None
            curr_mean = mean
            curr_stddev = stddev

        if return_force == False or return_force is None:
            self.derivative = None
            self.negative_dr = False
        else:
            self.derivative = 'dydx'
            self.negative_dr = True

        if return_contributions:
            contributions = 'yi'
        else:
            contributions = None


        super(MultiState, self).__init__(
            n_in,
            n_out=n_states,
            aggregation_mode=aggregation_mode,
            n_layers=n_layers,
            n_neurons=n_neurons,
            activation=activation,
            contributions=contributions,
            derivative=self.derivative,
            negative_dr=self.negative_dr,
            create_graph=create_graph,
            mean=curr_mean,
            stddev=curr_stddev,
            atomref=atomref,
            outnet=outnet
        )


    def forward(self, inputs):
        result = super(MultiState, self).forward(inputs)
        self.finish = self.return_hessian[5]
        if self.derivative:
            if self.n_states==int(1):
                i=0
                dydr = torch.stack([grad(result["y"][:, i], inputs[spk.Properties.R],
                                     grad_outputs=torch.ones_like(result["y"][:, i]),
                                     create_graph=self.create_graph,
                                     retain_graph=True)[0]], dim=1)
            if self.n_states > int(1):
                dydr = torch.stack([grad(result["y"][:, i], inputs[spk.Properties.R],
                                     grad_outputs=torch.ones_like(result["y"][:, i]),
                                     create_graph=self.create_graph,
                                     retain_graph=True)[0] for i in range(self.n_states)], dim=1)
            if self.n_states == int(0):
                dydr=0
            result['dydx'] = dydr
            if self.return_hessian[0]:
                n_singlets=self.return_hessian[1]
                n_triplets=self.return_hessian[3]
                threshold_dE_S=self.return_hessian[2]
                threshold_dE_T=self.return_hessian[4]
                compute_hessian=False
                for istate in range(n_singlets):
                    for jstate in range(istate+1,n_singlets):
                        if abs(result['y'][0][istate].item()-result['y'][0][jstate].item()) <= threshold_dE_S:
                            compute_hessian=True
                        else:
                            pass
                for istate in range(n_singlets,n_singlets+n_triplets):
                    for jstate in range(istate+1,n_singlets+n_triplets):
                        if abs(result['y'][0][istate].item()-result['y'][0][jstate].item()) <= threshold_dE_T:
                            compute_hessian=True
                        else:
                            pass
                #only for SO2
                #compute_hessian=False

                #if abs(result['y'][0][0].item()-result['y'][0][1].item()) <= threshold_dE_S or  abs(result['y'][0][0].item()-result['y'][0][2].item()) <= threshold_dE_S:
                #  compute_hessian=True
                #if abs(result['y'][0][3].item()-result['y'][0][5].item()) <= threshold_dE_T:
                #  compute_hessian=True
                if self.finish == True:
                    compute_hessian=False
                    if abs(result["y"][0][0].item()-result["y"][0][1].item()) <= threshold_dE_S:
                        compute_hessian=True
                batch, states, natoms, _ = dydr.shape
                # BEWARE: detach makes learning of hessians impossible, but saves a LOT of memory for prediction
                if compute_hessian==True:
                    d2ydr2 = torch.stack([grad(dydr.view(batch, -1)[:, i], inputs[spk.Properties.R],
                                           grad_outputs=torch.ones_like(dydr.view(batch, -1)[:, i]),
                                           create_graph=self.create_graph)[0].detach() for i in
                                      range(self.n_states * natoms * 3)], dim=1)

                    d2ydr2 = d2ydr2.view(batch, states, 3 * natoms, 3 * natoms)
                    result['d2ydx2'] = d2ydr2
                else:
                    #for scan enable those lines
                    #d2ydr2 = torch.stack([grad(dydr.view(batch, -1)[:, i], inputs[spk.Properties.R],
                    #                      grad_outputs=torch.ones_like(dydr.view(batch, -1)[:, i]),
                    #                      create_graph=self.create_graph)[0].detach() for i in
                    #                 range(self.n_states * natoms * 3)], dim=1)

                    #d2ydr2 = d2ydr2.view(batch, states, 3 * natoms, 3 * natoms)
                    d2ydr2 = torch.zeros([batch, states, 3 * natoms, 3 * natoms])
                    result['d2ydx2'] = d2ydr2
        return result



class MultiEnergy(MultiState):
    """
    Basic output module for energies and forces on multiple states
    """

    def __init__(self, n_in, n_states, aggregation_mode='sum', n_layers=2,
                 n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 return_contributions=False, create_graph=True,
                 return_force=False, mean=None, stddev=None, atomref=None,
                 max_z=100, outnet=None, return_hessian=[False,1,0,1,0,False], order=None,n_triplets = 0):
        super(MultiEnergy, self).__init__(n_in, n_states, aggregation_mode, n_layers,
                                          n_neurons, activation,
                                          return_contributions, create_graph,
                                          return_force, mean, stddev,
                                          atomref, max_z, outnet,
                                          return_hessian=return_hessian,order=order,n_triplets =n_triplets)

    #def __init__(self, n_in, n_states, aggregation_mode='sum', n_layers=2, n_neurons=None,
    #             activation=spk.nn.activations.shifted_softplus, return_contributions=False, create_graph=True,
    #             return_force=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None,
    #             standardize_after=False, return_hessian=[False,1,0.018,1,0.036]):

    def forward(self, inputs):
        """
        predicts energy
        """
        if self.return_hessian == False:
          self.return_hessian=[False,1,0,1,0,False]
        result = super(MultiEnergy, self).forward(inputs)

        # Apply negative gradient for forces
        if self.derivative:
            result['dydx'] = -result['dydx']

        #order energies according to energy
        #only for spin-diabatic states
        if self.order == True:
            if self.n_triplets == int(0):
                sort = torch.sort(result['y'])
                energies_singlets = sort[0]
                index_singlets = sort[1]
                result['y'] = energies_singlets
            else:
                print("order function not implemented for triplet energies")
                #todo 
                #needs to be adapted for triplets
                """self.n_singlets = result['y'].size()[1] - self.n_triplets
                index_singlets = torch.sort(result['y'][:,:self.n_singlets])[1]
                energies_singlets = torch.sort(result['y'][:,:self.n_singlets])[0]
                index_triplets = torch.sort(result['y'][:,self.n_singlets:])[1]
                energies_triplets = torch.sort(result['y'][:,self.n_singlets:])[0]
                result['y'][:,:self.n_singlets] = energies_singlets
                result['y'][:,self.n_singlets:] = energies_triplets
                #energies_singlets = torch.gather(result['y'][:,:self.n_singlets],1,index_singlets)"""
            if 'dydx' in result:
                #print(result['dydx'].size())
                if self.n_triplets == int(0):
                    #sort along first axis using indices from energy-sorting
                    d1,d2,d3,d4 = result['dydx'].size()
                    fi=torch.ones((d1,d2,d3,d4),device=result['y'].device)
                    fi=(fi*index_singlets[:,:,None,None]).long()
                    forces = torch.gather(result['dydx'],1,fi)
                    result['dydx'] = forces
                else:
                    print("order function not implemented for triplet forces")
                    #TODO
                    #needs to be adapted for triplets
                    """d1,d2,d3,d4 = result['dydx'].size()
                    fi=torch.ones((d1,d2,d3,d4),device=result['y'].device)
                    #todo make this work
                    fi=(fi*index_triplets[:,:,None,None]).long()
                    forces_singlets = torch.gather(result['dydx'][:,:self.n_singlets],1,fi)
                    f=result['dydx']
                    forces_triplets = torch.gather(result['dydx'][:,self.n_singlets:],1,index_triplets)
                    result['dydx'][:,:self.n_singlets] = forces_singlets
                    result['dydx'][:,self.n_singlets:] = forces_triplets"""
        return result


class MultiDipole(MultiState):
    """
        Predicts dipole moments.

        Args:
            n_in (int): input dimension of representation
            pool_mode (str): one of {sum, avg} (default: sum)
            n_layers (int): number of nn in output network (default: 2)
            n_neurons (list of int or None): number of neurons in each layer of the output network.
                                              If `None`, divide neurons by 2 in each layer. (default: none)
            activation (function): activation function for hidden nn (default: spk.nn.activations.shifted_softplus)
            return_contributions (bool): If True, latent atomic contributions are returned as well (default: False)
            create_graph (bool): if True, graph of forces is created (default: False)
            return_force (bool): if True, forces will be calculated (default: False)
            mean (torch.FloatTensor): mean of energy (default: None)
            stddev (torch.FloatTensor): standard deviation of the energy (default: None)
            atomref (torch.Tensor): reference single-atom properties
            outnet (callable): network used for atomistic outputs. Takes schnetpack input dictionary as input. Output is
                               not normalized. If set to None (default), a pyramidal network is generated automatically.

        Returns:
            tuple: Prediction for energy.

            If requires_dr is true additionally returns forces
        """

    def __init__(self, n_in, n_states, aggregation_mode='sum', n_layers=2, n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus, create_graph=True, outnet=None):

        n_singlets = n_states[Properties.n_singlets]
        n_triplets = n_states[Properties.n_triplets]
        nmstates   = n_singlets + 3 * n_triplets
        n_couplings = int(n_singlets*(n_singlets-1)/2 + n_singlets) + int(n_triplets*(n_triplets-1)/2 + n_triplets)
        #n_couplings = int(nmstates * (nmstates - 1) / 2 + nmstates)

        super(MultiDipole, self).__init__(n_in, n_couplings,
                                          aggregation_mode=aggregation_mode,
                                          n_layers=n_layers,
                                          n_neurons=n_neurons,
                                          activation=activation,
                                          return_contributions=True,
                                          create_graph=create_graph,
                                          atomref=None,
                                          max_z=100,
                                          outnet=outnet)

        #self.output_mask = schnarc.nn.DipoleMask( n_couplings,n_singlets=n_singlets,n_triplets=n_triplets)


    def forward(self, inputs):
        """
        predicts dipole moments
        uses a mask that values where no dipoles exist (e.g. singlet - triplets) are zero
        """
        result = super(MultiDipole, self).forward(inputs)
        dipole_moments = torch.sum(result["yi"][:, :, :, None] * inputs[spk.Properties.R][:, :, None, :], 1)
        #print(result['yi'][0])
        #print(inputs)
        #print(dipole_moments.shape,inputs[spk.Properties.R].shape)
        #print(result['yi'].shape)
        result['y'] = dipole_moments #self.output_mask(dipole_moments)

        return result


class MultiNac(MultiState):
    """
        Predicts nonadiabatic couplings - nstates  vectors of Natoms x 3.

        Args:
            n_in (int): input dimension of representation
            pool_mode (str): one of {sum, avg} (default: sum)
            n_layers (int): number of nn in output network (default: 2)
            n_neurons (list of int or None): number of neurons in each layer of the output network.
                                              If `None`, divide neurons by 2 in each layer. (default: none)
            activation (function): activation function for hidden nn (default: spk.nn.activations.shifted_softplus)
            return_contributions (bool): If True, latent atomic contributions are returned as well (default: False)
            create_graph (bool): if True, graph of forces is created (default: False)
            return_force (bool): if True, forces will be calculated (default: False)
            mean (torch.FloatTensor): mean of energy (default: None)
            stddev (torch.FloatTensor): standard deviation of the energy (default: None)
            atomref (torch.Tensor): reference single-atom properties
            outnet (callable): network used for atomistic outputs. Takes schnetpack input dictionary as input. Output is
                               not normalized. If set to None (default), a pyramidal network is generated automatically.

        Returns:
            tuple: Prediction for energy.

            If requires_dr is true additionally returns forces
        """
    def __init__(self, n_in, n_states, aggregation_mode='sum', n_layers=2, n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus, return_contributions=False, create_graph=True,
                 outnet=None, use_inverse=0):


        n_singlets = n_states[Properties.n_singlets]
        n_triplets = n_states[Properties.n_triplets]
        n_couplings = int(n_singlets * (n_singlets - 1) / 2 + n_triplets * (n_triplets - 1) / 2)

        super(MultiNac, self).__init__(n_in, n_couplings,
                                       aggregation_mode=aggregation_mode,
                                       n_layers=n_layers,
                                       n_neurons=n_neurons,
                                       activation=activation,
                                       return_contributions=return_contributions,
                                       create_graph=create_graph,
                                       return_force=True,
                                       atomref=None,
                                       max_z=100,
                                       outnet=outnet,)
                                       #use_inverse = use_inverse)

        self.use_inverse = use_inverse

        if self.use_inverse != 0:
            reconsts = n_singlets + n_triplets
            self.approximate_inverse = schnarc.nn.ApproximateInverse(reconsts,n_triplets=n_triplets)

    def forward(self, inputs):
        self.return_hessian=[False,1,0,1,0,False]
        result = super(MultiNac, self).forward(inputs)
        result['y2']=result['y']
        if ("energy" in inputs or "nac_energy" in inputs ) and  self.use_inverse != 0:
            if "energy" in inputs and "nac_energy" not in inputs:
                inputs["nac_energy"] = inputs["energy"]
            inv_ener = self.approximate_inverse(inputs['nac_energy'])
            #result['dydx_old'] = result['dydx']
            result['dydx'] = inv_ener[:, :, None, None] * result['dydx']
            result['y'] = inv_ener * result['y']
        return result


class MultiSoc(MultiState):
    """
        Predicts spinorbit couplings.

        Args:
            n_in (int): input dimension of representation
            pool_mode (str): one of {sum, avg} (default: sum)
            n_layers (int): number of nn in output network (default: 2)
            n_neurons (list of int or None): number of neurons in each layer of the output network.
                                              If `None`, divide neurons by 2 in each layer. (default: none)
            activation (function): activation function for hidden nn (default: spk.nn.activations.shifted_softplus)
            return_contributions (bool): If True, latent atomic contributions are returned as well (default: False)
            create_graph (bool): if True, graph of forces is created (default: False)
            return_force (bool): if True, forces will be calculated (default: False)
            mean (torch.FloatTensor): mean of energy (default: None)
            stddev (torch.FloatTensor): standard deviation of the energy (default: None)
            atomref (torch.Tensor): reference single-atom properties
            outnet (callable): network used for atomistic outputs. Takes schnetpack input dictionary as input. Output is
                               not normalized. If set to None (default), a pyramidal network is generated automatically.

        Returns:
            tuple: Prediction for energy.

            If requires_dr is true additionally returns forces
        """

    def __init__(self, n_in, n_states, aggregation_mode='sum', n_layers=2, n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus, create_graph=False, mean=None, stddev=None,
                 outnet=None, real=False):
        super(MultiSoc, self).__init__(n_in, n_states,
                                       aggregation_mode=aggregation_mode,
                                       n_layers=n_layers,
                                       n_neurons=n_neurons,
                                       activation=activation,
                                       return_contributions=False,
                                       create_graph=create_graph,
                                       mean=mean,
                                       stddev=stddev,
                                       atomref=None,
                                       max_z=100,
                                       outnet=outnet)

        self.output_mask = schnarc.nn.SocsTransform(n_states, real=real)

    def forward(self, inputs):
        """
        predicts spin-orbit couplings
        """

        result = super(MultiSoc, self).forward(inputs)

        socs = self.output_mask(result['y'])
        result['y'] = socs

        return result


class HiddenStatesEnergy(Atomwise):
    """
    TODO: This will never produce a diabatisation, as it is time-independent
    """

    def __init__(self, n_in, n_states, aggregation_mode='sum', n_layers=2,
                 n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 return_contributions=False, create_graph=True,
                 return_force=False, mean=None, stddev=None, atomref=None,
                 max_z=100, outnet=None):
        self.n_states = n_states

        mav = torch.mean(mean)
        msd = torch.mean(stddev)

        n_virt = 4 * n_states
        self.n_virt = n_virt

        # mean = torch.ones(n_virt) * mav  # - torch.rand(n_virt)
        # stddev = torch.ones(n_virt) * msd
        # mean = torch.cat((mean, mean))
        # stddev = torch.cat((stddev, stddev))

        super(HiddenStatesEnergy, self).__init__(n_in, n_virt ** 2, aggregation_mode, n_layers,
                                                 n_neurons, activation,
                                                 return_contributions, return_force,
                                                 create_graph, None, None,
                                                 atomref, max_z, outnet)

        self.standardnet = spk.nn.ScaleShift(mean, stddev)
        self.transformer = spk.nn.MLP(n_in, n_virt ** 2, n_hidden=2)
        self.classifyer = spk.nn.MLP(n_in, n_virt ** 2, n_hidden=2)

        self.diag_mask = torch.eye(self.n_virt).long()
        self.globalrep = schnarc.nn.GlobalRepresentation(n_in)

    def forward(self, inputs):
        """
        predicts energy
        """
        result = super(HiddenStatesEnergy, self).forward(inputs)

        B, A, _ = inputs[spk.Properties.R].shape

        global_representation = self.globalrep(inputs['representation'])

        # Diabatic matrix
        all_energies = result['y']
        all_energies = all_energies.view(B, self.n_virt, self.n_virt)
        all_energies = 0.5 * (all_energies + all_energies.permute(0, 2, 1))

        # TODO: Make global and add dense layer
        tmat = self.transformer(global_representation).view(B, self.n_virt, self.n_virt)
        # tmat = torch.nn.functional.softmax(tmat, dim=1)
        # tmat = tmat / torch.norm(tmat, 2, 2, keepdim=True)

        # Enforce orthogonality
        t_loss = (torch.matmul(tmat, tmat.permute(0, 2, 1)) - torch.eye(self.n_virt, device=all_energies.device)[None,
                                                              :, :]) ** 2

        result['tloss'] = t_loss
        #print(t_loss[0])
        #print(tmat[0])
        #print(torch.norm(tmat[0], 2, 0))
        #print(torch.norm(tmat[0], 2, 1))

        # Transform to adiabatic
        mixed_energies = torch.matmul(tmat.permute(0, 2, 1), all_energies)
        mixed_energies = torch.matmul(mixed_energies, tmat)

        # Define a off diagonal loss
        off_diag = mixed_energies[:, self.diag_mask != 1]
        result['off_diag'] = off_diag ** 2

        # Get energies
        mixed_energies = mixed_energies[:, self.diag_mask == 1]

        energies, states = torch.sort(mixed_energies, dim=1)
        # energies, states = torch.sort(all_energies, dim=1)

        # energies = self.standardnet(energies)
        # energies = self.standardnet(all_energies[:, self.diag_mask == 1])

        masked_energies = energies[:, :self.n_states]
        masked_energies = self.standardnet(masked_energies)
        virtual_energies = energies[:, self.n_states:]

        result['y_all'] = all_energies[:, self.diag_mask == 1]
        result['y_all_mixed'] = mixed_energies

        result['y'] = masked_energies
        result['y_virtual'] = virtual_energies
        result['c_states'] = states

        if self.requires_dr:
            nonadiabatic_couplings = torch.stack([-grad(result["y"][:, i], inputs[spk.Properties.R],
                                                        grad_outputs=torch.ones_like(result["y"][:, i]),
                                                        create_graph=self.create_graph)[0] for i in
                                                  range(self.n_states)], dim=1)
            result.update({'y': nonadiabatic_couplings})
        return result
