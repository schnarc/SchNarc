import schnetpack as spk
import torch
import torch.nn as nn

from torch.autograd import grad
from schnetpack import Properties
from schnetpack.atomistic import Atomwise
import schnet_ev
from schnet_ev.output_modules_spk_atomistic import AtomwiseH
from schnet_ev.data import Properties, Eigenvalue_properties, Delta_EV_properties


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
        # TODO: double for might not be the nicest. Worst case reigenvaluesert to preigenvaluesious dict of dict
        #   This version has advantage, that eigenvalueserthing is accessible in output and no parallel model collisions
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
    def __init__(self, n_in, n_orb, properties,n_neurons=None, mean=None, stddev=None, atomref=None, need_atomic=False, n_layers=2,
                 inverse_energy=False, diabatic=False, hamiltonian=False,force_mask = False, force_mask_index=None ):

        super(MultiStatePropertyModel, self).__init__()

        self.n_in = n_in
        self.properties = properties
        self.need_atomic = need_atomic
        self.diabatic = diabatic
        self.hamiltonian = hamiltonian
        self._init_property_flags(properties)
        self.n_orb = n_orb['n_orb']
        self.active = n_orb['n_act']
        self.n_occ=n_orb['n_occ']
        self.n_unocc = n_orb['n_unocc']
        self.aggregation_mode = n_orb["aggregation_mode"] 
        if "active_all" in n_orb:
            self.active_all = int((n_orb["active_all"]*(n_orb["active_all"]+1)/2))
        else:
            self.active_all = int((n_orb["n_act"]*(n_orb["n_act"]+1)/2))
        if self.need_forces is not None: #TODO Enable if necessary or self.need_eigenvalues_forces or self.need_eigenvalues_active_forces:
            self.derivative = 'dydx'
        else:
            self.derivative = None
            return_force=False
        # Construct default mean and stddevs if passed None
        if mean is None or stddev is None:
            mean = {p: None for p in properties}
            stddev = {p: None for p in properties}

        # Disable stress
        self.stress = False

        outputs = {}
        # Energies and forces
        if self.need_energy or self.need_forces and self.need_force_mask==False:
            try:
                atomref = atomref[Properties.energy]
            except:
                atomref = None

            n_ener=int(1) 
            energy_module = Energy(n_in, n_ener,  aggregation_mode='sum',
                                        return_force=self.need_forces,
                                        n_neurons=n_neurons,
                                        return_contributions=self.need_atomic, mean=mean[Properties.energy],
                                        stddev=stddev[Properties.energy],
                                        atomref=atomref, create_graph=True, n_layers=n_layers)

            outputs[Properties.energy] = energy_module
        if self.need_energy or self.need_forces and self.need_force_mask==True:
            #train on energy and forces with mask
            try:
                atomref=atomref[Properties.energy]
            except:
                atomref=None
            energy_module = Energy(n_in,n_orb=int(1), aggregation_mode="sum",
                                         return_force=self.need_forces,
                                         n_neurons=n_neurons,
                                         return_contributions=self.need_atomic, mean=mean[Properties.energy],
                                         stddev=stddev[Properties.energy],
                                         atomref=atomref, create_graph=True, n_layers=n_layers,force_mask=force_mask,force_mask_index=force_mask_index)
 
            outputs[Properties.energy] = energy_module

        #eigenvalues
        if self.need_occ_orb:
            eigenvalue_module = MultiOrb(n_in, self.n_occ, aggregation_mode='sum',
                                         return_force = self.need_forces,
                                         n_neurons=n_neurons,
                                         return_contributions=self.need_atomic, mean=mean[Properties.occ_orb],
                                         stddev=stddev[Properties.occ_orb],
                                         atomref=atomref, create_graph=True, n_layers=n_layers, diabatic=self.diabatic,hamiltonian=self.hamiltonian)
            outputs[Properties.occ_orb] = eigenvalue_module
            #outputs[Properties.occ_orb], outputs[Properties.homo_lumo], outputs[Properties.delta_E] = eigenvalue_module[0],eigenvalue_module[1], eigenvalue_module[2]
        if self.need_unocc_orb:
            eigenvalue_module = MultiOrb(n_in, self.n_unocc, aggregation_mode='sum',
                                         return_force = self.need_forces,
                                         n_neurons=n_neurons,
                                         return_contributions=self.need_atomic, mean=mean[Properties.unocc_orb],
                                         stddev=stddev[Properties.unocc_orb],
                                         atomref=atomref, create_graph=True, n_layers=n_layers, diabatic=self.diabatic,hamiltonian=self.hamiltonian)
            outputs[Properties.unocc_orb] = eigenvalue_module
 
        if self.need_delta_E:
            delta_E_module = delta_E(n_in, int(1),  aggregation_mode='sum',
                                        return_force=self.need_forces,
                                        n_neurons=n_neurons,
                                        return_contributions=self.need_atomic, mean=mean[Properties.delta_E],
                                        stddev=stddev[Properties.delta_E],
                                        atomref=atomref, create_graph=True, n_layers=n_layers)

            outputs[Properties.delta_E] = delta_E_module
 
        if self.need_homo_lumo:
             homo_lumo_module = Homo_Lumo(n_in, int(1),  aggregation_mode='sum',
                                        return_force=self.need_forces,
                                        n_neurons=n_neurons,
                                        return_contributions=self.need_atomic, mean=mean[Properties.homo_lumo],
                                        stddev=stddev[Properties.homo_lumo],
                                        atomref=atomref, create_graph=True, n_layers=n_layers)

             outputs[Properties.homo_lumo] = homo_lumo_module



        if self.delta_learning_ev:
            eigenvalue_module = MultiOrb(n_in, self.active, aggregation_mode='sum',
                                         return_force = self.need_forces,
                                         n_neurons=n_neurons,
                                         return_contributions=self.need_atomic,mean=mean[self.delta_learning_ev_prop],
                                         stddev=stddev[self.delta_learning_ev_prop],
                                         atomref=atomref, create_graph=True, n_layers=n_layers, diabatic=self.diabatic,hamiltonian=False)
            outputs[self.delta_learning_ev_prop] = eigenvalue_module
        if self.build_something:
            eigenvalue_module = PseudoH(n_in,(self.active,self.active_all), aggregation_mode=self.aggregation_mode,
                                         return_force = self.need_forces,
                                         n_neurons=n_neurons,
                                         return_contributions=self.need_atomic, mean=None,
                                         stddev=None,
                                         atomref=atomref, create_graph=True, n_layers=n_layers, diabatic=self.diabatic,hamiltonian=self.hamiltonian)
            outputs[self.need_eigenvalues_active_prop] = eigenvalue_module
        if self.build_hamiltonian:
            eigenvalue_module = PseudoH(n_in,self.active_all, aggregation_mode='build_atomic',
                                         return_force = self.need_forces,
                                         n_neurons=n_neurons,
                                         return_contributions=self.need_atomic, mean=None,
                                         stddev=None,
                                         atomref=atomref, create_graph=True, n_layers=n_layers, diabatic=self.diabatic,hamiltonian=self.hamiltonian)
            outputs[self.need_eigenvalues_active_prop] = eigenvalue_module
 
    
        self.output_dict = nn.ModuleDict(outputs)

    def _init_property_flags(self, properties):
        self.need_energy    = Properties.energy in properties
        if Properties.forces in properties:
            self.need_forces = True
        else:
            self.need_forces = False
        self.need_occ_orb   = Properties.occ_orb in properties
        self.need_unocc_orb = Properties.unocc_orb in properties
        self.need_delta_E   = Properties.delta_E in properties
        self.hamiltonian        = False # Properties.eigenvalues in properties #False # does not work but kept in case it will be needed sometime again, requires additional block for offdiagonal elements
        self.build_hamiltonian        = False # Properties.eigenvalues in properties #False # does not work but kept in case it will be needed sometime again, requires additional block for offdiagonal elements
        self.need_eigenvalues_forces = Properties.eigenvalues_forces in properties
        self.need_homo_lumo = Properties.homo_lumo in properties
        self.need_eigenvalues_active = False
        self.build_something = False
        self.delta_learning_ev = False
        self.need_eigenvalues_active_forces = False
        for prop in properties:
            if prop in Eigenvalue_properties.properties:
                self.build_something = True
                self.need_eigenvalues_active_prop = prop
            if prop in Delta_EV_properties.properties:
                self.delta_learning_ev= True
                self.delta_learning_ev_prop = prop
    def forward(self, inputs):

        outputs = {}
        for prop, model in self.output_dict.items():
            result = model(inputs)
            outputs[prop] = result['y']
            if prop == Properties.energy and self.need_forces:
                outputs[Properties.forces] = result['dydx']
            if prop == Properties.energy and 'd2ydx2' in result:
                outputs[Properties.hessian] = result['d2ydx2']
            if prop == Properties.eigenvalues:
                outputs['stdev'] = result['y_i']
            if prop == Properties.occ_orb and self.need_forces:
                outputs[Properties.occ_orb_forces] = result['dydx']
            if prop == Properties.eigenvalues_active and self.need_eigenvalues_active_forces==True or prop == Properties.eigenvalues and self.need_eigenvalues_forces==True:
                outputs[prop] = {}
                #uses a lot of memory
                #outputs[Properties.eigenvalues_active]['eigenvalues_active_forces'] = result['dydx']
                outputs[prop]['ev_diabatic'] = result['y_i']
                #outputs[prop]['ev_diab_forces'] = result['dydxi']
                outputs[prop][prop] = result['y']
            if self.diabatic:
              if prop ==  Properties.eigenvalues_active and self.need_eigenvalues_active_forces==False or prop ==  Properties.eigenvalues and self.need_eigenvalues_forces==False:
                outputs['ev_diabatic'] = result['y_i']
            #if self.hamiltonian:
            #    outputs['coeff']=result['coeff']
            #    outputs['pseudoH'] = result['y_i']
            if prop == "hirshfeld_pbe" or prop=="hirshfeld_pbe0" or prop=="hirshfeld_pbe0_h2o" or prop=="hirshfeld_h2o": #TODO activate for hirshfeld H
                pass
        return outputs


class MultiStateError(Exception):
    pass


class SingleState(Atomwise):

    def __init__(self, n_in, n_orb, aggregation_mode='sum', n_layers=2, n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus, return_contributions=False, create_graph=True,
                 return_force=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None,
                 standardize_after=False, force_mask = False, force_mask_index=None):

        if standardize_after and mean is not None and stddev is not None:
            self.standardize = spk.nn.ScaleShift(mean, stddev)
            curr_mean = None
            curr_stddev = None
        else:
            self.standardize = None
            curr_mean = mean
            curr_stddev = stddev

        if return_force == True: # or return_force is not None:
            self.derivative = 'dydx'
            self.negative_dr = True
        else:
            self.derivative = None
            self.negative_dr = False

        if return_contributions:
            contributions = 'yi'
        else:
            contributions = None

        self.force_mask_index = force_mask_index
        if force_mask:
            self.need_force_mask = True
        else:
            self.need_force_mask = False 

        super(SingleState, self).__init__(
            n_in,
            n_out=n_orb,
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
            outnet=outnet,
        )


    def forward(self, inputs):
        result = super(SingleState, self).forward(inputs)
        if self.derivative:
            if self.need_force_mask == True:
                    forces_index=(inputs['_atomic_numbers']!=int(self.force_mask_index)).nonzero().to(result[self.property].device)
                    #selects only inputs with indices
                    force_input = torch.index_select(inputs[spk.Properties.R],1,forces_index[:,1])
                    #print(force_input,inputs[spk.Properties.R][0][658:707])
                    #inputs[spk.Properties.R] = force_input
                    #print(inputs[spk.Properties.R])
                    dydr = grad(result[self.property],inputs[spk.Properties.R],grad_outputs=torch.ones_like(result[self.property]),create_graph=self.create_graph, retain_graph=True)[0]
                    zero_force = torch.zeros([inputs['_atomic_numbers'].shape[0],inputs['_atomic_numbers'].shape[1],3]).to(result[self.property].device)
                    dydr = torch.index_select(dydr,1,forces_index[:,1]) 
                    zero_force[:,forces_index[:,1],:] = dydr
                    result['dydx'] = zero_force
            else:
                    dydr = grad(result[self.property],inputs[spk.Properties.R],grad_outputs=torch.ones_like(result[self.property]),create_graph=self.create_graph, retain_graph=True)[0]
                    result['dydx']=dydr
        return result


class MultiH(AtomwiseH):

    def __init__(self, n_in, n_orb, aggregation_mode='sum', n_layers=2, n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus, return_contributions=False, create_graph=True,
                 return_force=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None,
                 standardize_after=False, return_hessian=[False,1,0.018,1,0.036], diabatic=False,hamiltonian=False):

        self.n_orb = n_orb
        self.hamiltonian=hamiltonian
        self.return_hessian = return_hessian
        self.diabatic = diabatic
        if self.diabatic or self.hamiltonian:
            self.return_contributions = True
            contributions = "yi"
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
        if return_force==True:
            if self.diabatic == False and mean is not None:
                self.derivative = 'dydx'
                self.negative_dr = True
            else:
                self.derivative = None
                self.negative_dr = False
        else:
            self.derivative = None
            self.negative_dr = False
        if self.hamiltonian: 
            return_force=False
        if return_contributions:
            contributions = 'yi'
        else:
            contributions = None
        super(MultiH, self).__init__(
            n_in,
            n_out=n_orb,
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
        result = super(MultiH, self).forward(inputs)
        return result

class MultiState(Atomwise):

    def __init__(self, n_in, n_orb, aggregation_mode='sum', n_layers=2, n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus, return_contributions=False, create_graph=True,
                 return_force=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None,
                 standardize_after=False, return_hessian=[False,1,0.018,1,0.036], diabatic=False,hamiltonian=False):

        self.n_orb = n_orb
        self.hamiltonian=hamiltonian
        self.return_hessian = return_hessian
        self.diabatic = diabatic
        if self.diabatic or self.hamiltonian:
            self.return_contributions = True
            contributions = "yi"
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
        if return_force==True:
            if self.diabatic == False and mean is not None:
                self.derivative = 'dydx'
                self.negative_dr = True
            else:
                self.derivative = None
                self.negative_dr = False
        else:
            self.derivative = None
            self.negative_dr = False
        if self.hamiltonian: 
            return_force=False
        if return_contributions:
            contributions = 'yi'
        else:
            contributions = None
        super(MultiState, self).__init__(
            n_in,
            n_out=n_orb,
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
        if self.derivative is not None:
            if self.n_orb==int(1):
                i=0
                #if self.need_force_mask == True:
                #    force_idx  = inputs[Properties.force_indices][0,:].lonag()
                #    force_input = torch.index_select(inputs[spk.Properties.R],1,force_idx)
                #    force_input.requires_grad = True
                #    inputs[spk.Properties.R][:,force_idx,:] = force_input
                #    dydr = grad(result[self.property],inputs['displacement'],grad_outputs=torch.ones_like(result[self.property]),create_graph=self.create_graph, retain_graph=True)[0]
                #else:
                dydr = torch.stack([grad(result["y"][:, i], inputs[spk.Properties.R],
                                     grad_outputs=torch.ones_like(result["y"][:, i]),
                                     create_graph=self.create_graph,
                                     retain_graph=True)[0]], dim=1)
            if self.n_orb > int(1):
                dydr = torch.stack([grad(result["y"][:, i], inputs[spk.Properties.R],
                                     grad_outputs=torch.ones_like(result["y"][:, i]),
                                     create_graph=self.create_graph,
                                     retain_graph=True)[0] for i in range(self.n_orb)], dim=1)
            result['dydx']=dydr 
        return result

class Energy(SingleState):
    """
    Basic output module for energies and forces on multiple states
    """

    def __init__(self, n_in, n_orb, aggregation_mode='sum', n_layers=2,
                 n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 return_contributions=False, create_graph=True,
                 return_force=False, mean=None, stddev=None, atomref=None,
                 max_z=100, outnet=None, force_mask  = False, force_mask_index=None):
        super(Energy, self).__init__(n_in, n_orb, aggregation_mode, n_layers,
                                          n_neurons, activation,
                                          return_contributions, return_force,
                                          create_graph, mean, stddev,
                                          atomref, max_z, outnet,
                                          force_mask = force_mask,
                                          force_mask_index  = force_mask_index )


    def forward(self, inputs):
        """
        predicts energy
        """
        result = super(Energy, self).forward(inputs)

        # Apply negative gradient for forces
        if self.derivative is not None:
            result['dydx'] = -result['dydx']
            
        return result


class MultiEnergy(MultiState):
    """
    Basic output module for energies and forces on multiple states
    """

    def __init__(self, n_in, n_orb, aggregation_mode='sum', n_layers=2,
                 n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 return_contributions=False, create_graph=True,
                 return_force=False, mean=None, stddev=None, atomref=None,
                 max_z=100, outnet=None, return_hessian=[False,1,0,1,0],diabatic=False,hamiltonian=False):
        super(MultiEnergy, self).__init__(n_in, n_orb, aggregation_mode, n_layers,
                                          n_neurons, activation,
                                          return_contributions, return_force,
                                          create_graph, mean, stddev,
                                          atomref, max_z, outnet,
                                          return_hessian=return_hessian,diabatic=diabatic,hamiltonian=hamiltonian)

    #def __init__(self, n_in, n_orb, aggregation_mode='sum', n_layers=2, n_neurons=None,
    #             activation=spk.nn.activations.shifted_softplus, return_contributions=False, create_graph=True,
    #             return_force=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None,
    #             standardize_after=False, return_hessian=[False,1,0.018,1,0.036]):

    def forward(self, inputs):
        """
        predicts energy
        """
        result = super(MultiEnergy, self).forward(inputs)

        # Apply negative gradient for forces
        if self.derivative is not None:
            result['dydx'] = -result['dydx']
        return result

class delta_E(MultiState):
    """
    Basic output module for energies and forces on multiple states
    """

    def __init__(self, n_in, n_orb, aggregation_mode='sum', n_layers=2,
                 n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 return_contributions=False, create_graph=True,
                 return_force=False, mean=None, stddev=None, atomref=None,
                 max_z=100, outnet=None, return_hessian=[False,1,0,1,0],diabatic=False,hamiltonian=False):
        super(delta_E, self).__init__(n_in, n_orb, aggregation_mode, n_layers,
                                          n_neurons, activation,
                                          return_contributions, return_force,
                                          create_graph, mean, stddev,
                                          atomref, max_z, outnet,
                                          return_hessian=return_hessian,diabatic=diabatic,hamiltonian=hamiltonian)


    def forward(self, inputs):
        """
        predicts energy
        """
        result = super(delta_E, self).forward(inputs)

        # Apply negative gradient for forces
        if self.derivative is not None:
            result['dydx'] = -result['dydx']

        return result


class Homo_Lumo(MultiState):
    """
    Basic output module for energies and forces on multiple states
    """

    def __init__(self, n_in, n_orb, aggregation_mode='sum', n_layers=2,
                 n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 return_contributions=False, create_graph=True,
                 return_force=False, mean=None, stddev=None, atomref=None,
                 max_z=100, outnet=None, return_hessian=[False,1,0,1,0],diabatic=False,hamiltonian=False):
        super(Homo_Lumo, self).__init__(n_in, n_orb, aggregation_mode, n_layers,
                                          n_neurons, activation,
                                          return_contributions, return_force,
                                          create_graph, mean, stddev,
                                          atomref, max_z, outnet,
                                          return_hessian=return_hessian,diabatic=diabatic,hamiltonian=hamiltonian)


    def forward(self, inputs):
        """
        predicts energy
        """
        result = super(Homo_Lumo, self).forward(inputs)

        # Apply negative gradient for forces
        if self.derivative is not None:
            result['dydx'] = -result['dydx']

        return result

class SumEV(MultiState):
    """
    Basic output module for energies and forces on multiple states
    """

    def __init__(self, n_in, n_occ, aggregation_mode='build', n_layers=3,
                 n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 return_contributions=False, create_graph=True,
                 return_force=False, mean=None, stddev=None, atomref=None,
                 max_z=100, outnet=None, return_hessian=[False,1,0,1,0], diabatic = False,hamiltonian=False):
        self.active=n_occ
        n_occ = int(n_occ*n_occ)
        super(SumEV, self).__init__(n_in, n_occ, aggregation_mode, n_layers,
                                          n_neurons, activation,
                                          return_contributions, return_force,
                                          create_graph, mean, stddev,
                                          atomref, max_z, outnet,
                                          return_hessian=return_hessian, diabatic=diabatic,hamiltonian=hamiltonian)

    def forward(self, inputs):
        """
        predicts energy
        """
        result = super(MultiState,self).forward(inputs)
        self.device = result['y'].device
        ev =torch.zeros(result['y'].size()[0],self.active,device=self.device)
        stdev =torch.zeros(result['y'].size()[0],self.active,device=self.device)
        index=0
        #predict more values than necessary and sum them up or average
        for i in range(self.active): 
            ev[:,i] = torch.mean(result['y'][:,index:(index+self.active)])
            stdev[:,i] = torch.std(result['y'][:,index:(index+self.active)])
            index+=self.active
        result['y_i'] = stdev #result['y']
        result['y']=ev
        result['coeff']= None

        # Apply negative gradient for forces
        if self.derivative is not None:
            result['dydx'] = -result['dydx']
        return result

class PseudoH(MultiH):
    """
    Basic output module for energies and forces on multiple states
    """

    def __init__(self, n_in, n_occ, aggregation_mode='build', n_layers=3,
                 n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 return_contributions=False, create_graph=True,
                 return_force=False, mean=None, stddev=None, atomref=None,
                 max_z=100, outnet=None, return_hessian=[False,1,0,1,0], diabatic = False,hamiltonian=False):
        self.aggregation_mode=aggregation_mode 
        self.active = n_occ[0]
        self.diabatic = diabatic
        if self.diabatic == True:
            n_orb=n_occ[1]
        else:
            n_orb = n_occ[0]
        super(PseudoH, self).__init__(n_in, n_orb, aggregation_mode, n_layers,
                                          n_neurons, activation,
                                          return_contributions, return_force,
                                          create_graph, mean, stddev,
                                          atomref, max_z, outnet,
                                          return_hessian=return_hessian, diabatic=diabatic,hamiltonian=hamiltonian)

    def forward(self, inputs):
        """
        predicts energy
        """
        
        if self.hamiltonian and self.aggregation_mode=="build_atomic":
            result = super(MultiH,self).forward(inputs)
            self.device = result['y'].device
            #diab_H = torch.bmm(diab_H,torch.transpose(diab_H,1,2))
            diabatic_values = result['y']
            all_andlumo = int(result['y'].shape[2]/2+1)
            diab_H =diabatic_values * torch.transpose(diabatic_values,1,2)
            eigenvalues,eigenvectors = torch.symeig(diab_H,eigenvectors=True)
            result['y'] = eigenvalues[:,:self.homolumo]
            result['coeff']=eigenvectors
            result['y_i'] = diab_H
        elif self.hamiltonian and self.aggregation_mode=="build":
            result = super(MultiH,self).forward(inputs)
            self.device = result['y'].device
            diabatic_values = result['y']
            diab_H = torch.zeros(diabatic_values.size()[0],result["n_orbs"],result["n_orbs"],device=self.device)
            diag=diab_H
            #TODO make this better but don't know how
            index=-1
            for i in range(result["n_orbs"]):
                index+=1
                diab_H[:,i,i] = diabatic_values[:,index]
                for j in range(i+1,result["n_orbs"]):
                    index+=1
                    diab_H[:,i,j] = diabatic_values[:,index]
                    diab_H[:,j,i] = diabatic_values[:,index]
             

            eigenvalues,eigenvectors = torch.symeig(diab_H,eigenvectors=True)
            result['y'] = eigenvalues[:,:result["n_orbs"]]
            result['coeff']=eigenvectors[:,:result["n_orbs"],:result["n_orbs"]]
            result['y_i'] = diab_H

        elif self.diabatic and self.aggregation_mode=="build" or self.diabatic and self.aggregation_mode == "avg":
            result = super(MultiH,self).forward(inputs)
            self.device = result['y'].device
            diabatic_values = result['y']
            diab_H = torch.zeros(diabatic_values.size()[0],self.active,self.active,device=self.device)
            diag=diab_H
            index=-1
            for i in range(self.active):
                index+=1
                diab_H[:,i,i] = diabatic_values[:,index]
                for j in range(i+1,self.active):
                    index+=1
                    diab_H[:,i,j] = diabatic_values[:,index]
                    diab_H[:,j,i] = diabatic_values[:,index]
             
            eigenvalues,eigenvectors = torch.symeig(diab_H,eigenvectors=True)
            result['y'] = eigenvalues
            result['coeff']=eigenvectors 
            result['y_i'] = diab_H

        else:
            result = super(PseudoH, self).forward(inputs)

        # Apply negative gradient for forces
        if self.derivative is not None:
            result['dydx'] = -result['dydx']
        return result

class MultiOrb(MultiState):
    """
    Basic output module for energies and forces on multiple states
    """

    def __init__(self, n_in, n_occ, aggregation_mode='sum', n_layers=3,
                 n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 return_contributions=False, create_graph=True,
                 return_force=False, mean=None, stddev=None, atomref=None,
                 max_z=100, outnet=None, return_hessian=[False,1,0,1,0], diabatic = False,hamiltonian=False):
        self.active=n_occ
        #index_mat = torch.zeros(self.active,self.active)
        #for i in range(self.active):
        #    index_mat[i,i:self.active] = 1 
        #self.index_mat = index_mat
        if diabatic == True: 
            self.inputsize = int(n_occ*(n_occ+1)/2) 
        elif hamiltonian == True:
            self.inputsize = int(n_occ*(n_occ+1)/2)
        else:
            self.inputsize = self.active
        super(MultiOrb, self).__init__(n_in, self.inputsize, aggregation_mode, n_layers,
                                          n_neurons, activation,
                                          return_contributions, return_force,
                                          create_graph, mean, stddev,
                                          atomref, max_z, outnet,
                                          return_hessian=return_hessian, diabatic=diabatic,hamiltonian=hamiltonian)

    def forward(self, inputs):
        """
        predicts energy
        """
        if self.diabatic:
            result = super(MultiOrb,self).forward(inputs)
            n_diab = result['y'].size()[1]
            self.device = result['y'].device
            if self.derivative is not None:
                diabatic_forces = result['dydx']
                diab_forces = torch.zeros(diabatic_values.size()[0],self.active,self.active,diabatic_forces.size()[2],diabatic_forces.size()[3],device=self.device)
                for i in range(self.active):
                     index+=1
                     diab_forces[:,i,i] = diabatic_forces[:,index]
                     for j in range(i+1,self.active):
                         index+=1
                         diab_forces[:,i,j] = diabatic_forces[:,index]
                         diab_forces[:,j,i] = -diabatic_forces[:,index]
                result['dydxi']=-diab_forces
            diabatic_values = result['y']
            diab_H = torch.zeros(diabatic_values.size()[0],self.active,self.active,device=self.device)
            diag=diab_H
            #TODO make this better but don't know how
            index=-1
            for i in range(self.active):
                index+=1
                diab_H[:,i,i] = diabatic_values[:,index]
                for j in range(i+1,self.active):
                    index+=1
                    diab_H[:,i,j] = diabatic_values[:,index]
                    diab_H[:,j,i] = diabatic_values[:,index]
            #diagonalize
            eigenvalues,eigenvectors = torch.symeig(diab_H,eigenvectors=True)
            #a=torch.mm(torch.mm(eigenvectors[0],torch.diag(eigenvalues[0])),eigenvectors[0].T)
            #b = torch.mm(torch.mm(eigenvectors[0].T,diab_H[0]),eigenvectors[0])
            #print(a)
            #print(diab_H[0])
            #print(b,eigenvalues[0])
            #DIAGONALIZATION
            #print(torch.mm(torch.mm(eigenvectors[0],eigenvalues[0]),eigenvectors[0].T))
            #U,eigenvalues,V= torch.svd(diab_H)
            #a=torch.mm(torch.mm(U[0].T,diab_H[0]),V[0])
            #print(a-torch.diag(eigenvalues[0]))
            #test
            #d=diag
            #a=diab_forces[:,self.active,self.active,3]
            ##print(torch.sub(torch.mm(diab_H[0].T,diab_H[0]),torch.mm(diab_H[0],diab_H[0].T)))
            #for batches in range(diabatic_values.size()[0]):
            #    diag[batches]=torch.diag(eigenvalues[batches])
            #    d[batches]=torch.matmul(torch.matmul(U[batches],diag[batches]),U[batches].T)
            #    a[batches]
            ##diab_H2 = torch.mm(torch.mm(U[0].T,diab_H[0]),U[0])
            #print(d-diab_H,d)
            #self.n_elec=torch.round(torch.sum((inputs['_atomic_numbers']),dim=1)*0.5+2)
            #make a mask and account only for homo+lumo+1
            #elec_mask = self.get_electron_mask()
            result['y']=eigenvalues
            result['y_i'] = diab_H
        elif self.hamiltonian:
            result = super(MultiOrb,self).forward(inputs)
            n_diab = result['y'].size()[1]
            self.device = result['y'].device
            diabatic_values = result['y']
            diab_H = torch.zeros(diabatic_values.size()[0],self.active,self.active,device=self.device)
            diag=diab_H
            #TODO make this better but don't know how
            index=-1
            for i in range(self.active):
                index+=1
                diab_H[:,i,i] = diabatic_values[:,index]
                for j in range(i+1,self.active):
                    index+=1
                    diab_H[:,i,j] = diabatic_values[:,index]
                    diab_H[:,j,i] = diabatic_values[:,index]
            #diagonalize
            eigenvalues,eigenvectors = torch.symeig(diab_H,eigenvectors=True)
            result['y']=eigenvalues
            result['coeff']=eigenvectors
            result['y_i'] = diab_H
        else:
            result = super(MultiOrb, self).forward(inputs)

        # Apply negative gradient for forces
        if self.derivative is not None:
            result['dydx'] = -result['dydx']
        return result
    
    def get_electron_mask(self):
        #odd number of electrons: lower number as 
        mask=torch.zeros(len(self.n_elec),self.active,self.active,device=self.device)
        #TODO mask noeed to be done 
        rows=torch.arange(self.n_elec.long(),out=torch.LongTensor())
        print(rows)
        mask[:self.n_elec.long()]= int(1)
        print(mask)
        return mask
        
class MultiEigenvalue(MultiState):
    """
    Basic output module for energies and forces on multiple states
    """

    def __init__(self, n_in, n_occ, aggregation_mode='sum', n_layers=2,
                 n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 return_contributions=False, create_graph=True,
                 return_force=False, mean=None, stddev=None, atomref=None,
                 max_z=100, outnet=None, return_hessian=[False,1,0,1,0],diabatic=False,hamiltonian=False):
        super(MultiEigenvalue, self).__init__(n_in, n_occ, aggregation_mode, n_layers,
                                          n_neurons, activation,
                                          return_contributions, return_force,
                                          create_graph, mean, stddev,
                                          atomref, max_z, outnet,
                                          return_hessian=return_hessian,diabatic=diabatic,hamiltonian=hamiltonian)

    #def __init__(self, n_in, n_orb, aggregation_mode='sum', n_layers=2, n_neurons=None,
    #             activation=spk.nn.activations.shifted_softplus, return_contributions=False, create_graph=True,
    #             return_force=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None,
    #             standardize_after=False, return_hessian=[False,1,0.018,1,0.036]):

    def forward(self, inputs):
        """
        predicts energy
        """
        result = super(MultiEigenvalue, self).forward(inputs)

        # Apply negative gradient for forces
        if self.derivative is not None:
            result['dydx'] = -result['dydx']

        #return result_occ, result_homo_lumo, result_deltaE
        return result
