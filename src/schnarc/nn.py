import schnetpack as spk
import torch.nn as nn
import torch
import numpy as np


def force_weigth(target_energies, energy_cutoff=0.02):
    # Get the differences
    delta_e = (target_energies[:, :, None] - target_energies[:, None, :]) ** 2

    # Mask the same states
    delta_e[:, torch.eye(target_energies.shape[1]) == 1] = 1.0

    # Weighted
    weigths = 1.0 - spk.nn.cutoff.cosine_cutoff(delta_e, energy_cutoff)
    weigths = torch.prod(weigths, -1)[:, :, None, None]
    # delta_e = torch.mean(delta_e, -1)
    # weigths = 1.0 - spk.nn.cosine_cutoff(delta_e, energy_cutoff)
    return weigths.detach()


def phaseless_loss(target, predicted, dim=-1, eps=1e-6):
    # TODO: Move sum and logging out of loss
    norm_target = torch.norm(target, 2, dim, keepdim=True)
    norm_predicted = torch.norm(predicted, 2, dim, keepdim=True)

    loss_length = (norm_target - norm_predicted) ** 2
    loss_length = torch.mean(loss_length.view(-1))

    overlap = torch.sum(target * predicted / (norm_target * norm_predicted + eps), dim)
    loss_overlap = 1.0 - overlap ** 2
    loss_overlap = torch.mean(loss_overlap.view(-1))

    #print(loss_length, loss_overlap, 'length/overlap')

    loss = loss_length + loss_overlap
    return loss


def diagonal_phaseloss(target, predicted, n_states,device,single):
    socs_target = target['socs']
    dtype = torch.FloatTensor
    socs_predicted = predicted['socs'].to('cpu').detach().numpy()
    #this is a torch tensor:
    #socs_predicted = predicted['socs']
    diagonal_target = target['diagonal_energies']
    energy_predicted = predicted['energy'].to('cpu').detach().numpy()
    #this would be a torch tensor
    #energy_predicted = predicted['energy']
    nmstates=n_states['n_singlets']+3*n_states['n_triplets']
    #torch tensor
    #hamiltonian_full = torch.zeros(energy_predicted.shape[0],nmstates,nmstates).type(dtype).to(device)
    hamiltonian_full = np.zeros((energy_predicted.shape[0],nmstates,nmstates),dtype=complex)

    #energies
    all_energies = np.zeros((energy_predicted.shape[0],nmstates))
    #torch tensor
    #all_energies = torch.zeros(energy_predicted.shape[0],nmstates).type(dtype).to(device)
    all_energies[:,:n_states['n_singlets']+n_states['n_triplets']] = energy_predicted[:,:]
    all_energies[:,n_states['n_singlets']+n_states['n_triplets']:n_states['n_singlets']+n_states['n_triplets']*2] = energy_predicted[:,n_states['n_singlets']:]
    all_energies[:,n_states['n_singlets']+n_states['n_triplets']*2:n_states['n_singlets']+n_states['n_triplets']*3] = energy_predicted[:,n_states['n_singlets']:]

    #socs
    socs_complex = np.zeros((energy_predicted.shape[0], socs_predicted.shape[1]),dtype = complex)
    #socs_complex = np.zeros((energy_predicted.shape[0], socs_predicted.shape[1]),dtype = complex)
    for isoc in range(int(socs_predicted.shape[1]/2)):
        #torch does not support complex numbers 
        socs_complex[:,isoc]=np.real(socs_predicted[:,2*isoc])+(socs_predicted[:,2*isoc+1]*1j)
    iterator=-1
    for istate in range(nmstates):
        for jstate in range(istate+1,nmstates):
            iterator+=1
            hamiltonian_full[:,istate,jstate] = socs_complex[:,iterator]
    for i in range(hamiltonian_full.shape[0]):
        hamiltonian_full[i] = hamiltonian_full[i]+hamiltonian_full[i].conj().T
        np.fill_diagonal(hamiltonian_full[i],all_energies[i])

    #get diagonal
    eigenvalues,vec = np.linalg.eigh(hamiltonian_full)
    eigenvalues = torch.from_numpy(eigenvalues).to(device)
    diff_diag = (diagonal_target - eigenvalues ) ** 2
    if single==True:
        loss = torch.mean(diff_diag.view(-1)).to(device)
    else:
        diff_a = ( socs_target - predicted['socs'] ) ** 2
        diff_b = ( socs_target + predicted['socs'] ) ** 2
        diff_soc = torch.min(diff_a,diff_b)
        loss = 0.5 * torch.mean(diff_soc.view(-1)) + 0.5 * torch.mean(diff_diag.view(-1)).to(device)

    return loss

def min_loss_single_old(target, predicted, smooth=True, smooth_nonvec=False, loss_length=True):
    diff_a = (target - predicted) ** 2
    diff_b = (target + predicted) ** 2

    # Correct phase loss
    if smooth:
        reduce_dims = predicted.shape[:2] + (-1,)
        diff_a = diff_a.view(reduce_dims)
        diff_b = diff_b.view(reduce_dims)
        a = torch.mean(diff_a, 2, keepdim=True)
        b = torch.mean(diff_b, 2, keepdim=True)
        z = a + b + 1e-6
        coeff_a = b / z
        coeff_b = a / z
        diff = coeff_a * diff_a + coeff_b * diff_b
        #print(torch.mean(coeff_a), torch.mean(coeff_b), 'A/B')
    else:
        diff = torch.min(diff_a,diff_b)

    if smooth_nonvec:
        # don't take the mean of non-vectorial properties - those are from different states
        z = diff_a + diff_b + 1e-6
        coeff_a = diff_b / z
        coeff_b = diff_a / z
        diff = coeff_a * diff_a + coeff_b * diff_b
    loss = torch.mean(diff.view(-1))

    if loss_length:
        norm_target = torch.norm(target, 2, -1, keepdim=True)
        norm_predicted = torch.norm(predicted, 2, -1, keepdim=True)
        loss_length = (norm_target - norm_predicted) ** 2
        loss_length = torch.mean(loss_length.view(-1))
        #print(loss_length, 'LL')
        loss = loss + loss_length

    return loss



def min_loss_single(target, predicted, combined_phaseless_loss, n_states, props_phase, phase_vector_nacs = None, dipole=False):
    #phases of socs nacs and dipoles corrected independently -- carried out if only one of those properties trained for dynamics
    #computes all possibilities how the phase can arrange
    #takes the min of those possible combinations
    #for dynamics: NN produces a smooth potential hence phase is consistent along trajectory
    #number of couplings
    #number of possibilities of phase switches (n_states are the number of S + T states)
    n_phases = props_phase[0]
    batch_size = props_phase[1]
    device = props_phase[2]
    # valid if either nacs or socs are treated
    #one degree of freedom: set sign of first state to +1
    phase_pytorch = props_phase[3]
    batch_loss = torch.Tensor(predicted.size()).to(device)

    if dipole == False:
        pred_phase_vec = torch.Tensor(predicted.size()).to(device)

        n_socs = props_phase[4]
        all_states = props_phase[5]
        socs_phase_matrix = props_phase[6]
        #has the same dimension as pred_phase_vec
        phaseless_loss_all = torch.abs(target[0]-pred_phase_vec[0])
        # do this for each value of the mini batch separately
        for index_batch_sample in range(batch_size):
            phaseless_loss = float('inf')
            for index_phase_vector in range(n_phases):

                #multiply with each possible phase vector

                pred_phase_vec[index_batch_sample] = predicted[index_batch_sample] * socs_phase_matrix[index_phase_vector]
                diff = torch.abs((target[index_batch_sample]-pred_phase_vec[index_batch_sample]))
                diff_mean = torch.mean(diff.view(-1))
                if diff_mean < phaseless_loss:
                    phaseless_loss = diff_mean
                    phaseless_loss_all = diff
            batch_loss[index_batch_sample] = phaseless_loss_all

    else:
        n_dipoles = 3*int(predicted.size()[1])
        pred_phase_vec = torch.Tensor(batch_size,n_dipoles).to(device)
        #dipole has 3 components (x,y, and z)
        dipole_phase_matrix = torch.cat((props_phase[8],props_phase[8],props_phase[8]),0).view(n_phases,n_dipoles)
        phaseless_loss_all = torch.abs(target[0]-pred_phase_vec[0].view(predicted.size()[1],3))
        # do this for each value of the mini batch separately
        for index_batch_sample in range(batch_size):
            phaseless_loss = float('inf')
            for index_phase_vector in range(n_phases):

                #multiply with each possible phase vector
                pred_phase_vec[index_batch_sample] = predicted[index_batch_sample].view(1,n_dipoles) * dipole_phase_matrix[index_phase_vector]
                diff = torch.abs((target[index_batch_sample].view(1,n_dipoles)-pred_phase_vec[index_batch_sample]))
                diff_mean = torch.mean(diff.view(-1))
                if diff_mean < phaseless_loss:
                    phaseless_loss = diff_mean
                    phaseless_loss_all = diff
            batch_loss[index_batch_sample] = phaseless_loss_all.view(predicted.size()[1],3)


    return batch_loss


def min_loss(target, predicted, combined_phaseless_loss, n_states, props_phase, phase_vector_nacs = None, dipole=False):
    #phases of socs and nacs not allowed to be independent for dynamics
    #computes all possibilities how the phase can arrange
    #takes the min of those possible combinations
    #for dynamics: NN produces a smooth potential hence phase is consistent along trajectory
    #number of couplings
    #number of possibilities of phase switches (n_states are the number of S + T states)
    n_phases = props_phase[0]
    batch_size = props_phase[1]
    device = props_phase[2]
    pred_phase_vec = torch.Tensor(predicted.size()).to(device)

    if combined_phaseless_loss == True:
      if dipole == False:
          n_socs = props_phase[4]
          all_states = props_phase[5]
          socs_phase_matrix_1 = props_phase[6]
          socs_phase_matrix_2 = props_phase[7]
          #has the same dimension as pred_phase_vec
          diff_1 =  torch.Tensor(predicted.size()).to(device)
          diff_2 =  torch.Tensor(predicted.size()).to(device)
          # compute min loss
          #iterator over all sample and take the corresponding phase vector from all possible solutions within the "socs_phase_matrices"
          for index_batch_sample in range(batch_size):
              diff_1[index_batch_sample,:] = (target[index_batch_sample,:] - predicted[index_batch_sample,:] * socs_phase_matrix_1[int(phase_vector_nacs[index_batch_sample]),:])**2
              diff_2[index_batch_sample,:] = (target[index_batch_sample,:] - predicted[index_batch_sample,:] * socs_phase_matrix_2[int(phase_vector_nacs[index_batch_sample]),:])**2
          diff_1 = torch.mean(diff_1, 1, keepdim=True)
          diff_2 = torch.mean(diff_2, 1, keepdim=True)
          diff = torch.min(diff_1,diff_2)
      else:
          #props_phase[8] and [9] are the two possible combinations of phase matrices
          #take the phase matrix that was best for nacs
          #iterate over all samples in the mini batch

          n_dipoles = predicted.size()[1]
          diff_1 = torch.Tensor(batch_size,int(3*n_dipoles)).to(device)
          diff_2 = torch.Tensor(batch_size,int(3*n_dipoles)).to(device)
          for index_batch_sample in range(batch_size):
              #index for phasevector
              i = int(phase_vector_nacs[index_batch_sample])
              dipole_phase_matrix_1 = torch.cat([props_phase[8][i],props_phase[8][i],props_phase[8][i]],0)
              dipole_phase_matrix_2 = torch.cat([props_phase[9][i],props_phase[9][i],props_phase[9][i]],0)
              pred_1 = predicted[index_batch_sample].view(1,int(3*n_dipoles)) * dipole_phase_matrix_1
              pred_2 = predicted[index_batch_sample].view(1,int(3*n_dipoles)) * dipole_phase_matrix_2
              target_sample = target[index_batch_sample].view(1,int(3*n_dipoles))
              diff_1[index_batch_sample] = ( target_sample[0] - pred_1[0] ) **2
              diff_2[index_batch_sample] = ( target_sample[0] - pred_2[0] ) **2
          diff_1 = torch.mean(diff_1, 1, keepdim=True)
          diff_2 = torch.mean(diff_2, 1, keepdim=True)
          diff = torch.min(diff_1,diff_2)
      return diff

    else:
      # valid if either nacs or socs are treated
      #one degree of freedom: set sign of first state to +1
      phase_pytorch = props_phase[3]
      batch_loss = torch.Tensor(target.size()).to(device)
      phase_ordering=torch.Tensor(batch_size).to(device)
      phase_ordering[:] = 0
      phaseless_loss_all = torch.abs(target[0]-pred_phase_vec[0])
      # do this for each value of the mini batch separately
      for index_batch_sample in range(batch_size):
          phaseless_loss = float('inf')
          for index_phase_vector in range(n_phases):

              #multiply singlets and triplets separately with according values of phase
              coupling_iterator=0
              for i_singlet in range(n_states['n_singlets']):
                  for j_singlet in range(i_singlet+1,n_states['n_singlets']):
                      pred_phase_vec[index_batch_sample,coupling_iterator]=( predicted[index_batch_sample,coupling_iterator] * phase_pytorch[i_singlet,index_phase_vector] * phase_pytorch[j_singlet,index_phase_vector] )
                      coupling_iterator+=1

              for i_triplet in range(n_states['n_singlets'],n_states['n_singlets']+n_states['n_triplets']):
                  for j_triplet in range(i_triplet+1,n_states['n_singlets']+n_states['n_triplets']):
                      pred_phase_vec[index_batch_sample,coupling_iterator]=( predicted[index_batch_sample,coupling_iterator] * phase_pytorch[i_triplet,index_phase_vector] * phase_pytorch[j_triplet,index_phase_vector] )

              diff = torch.abs(target[index_batch_sample]-pred_phase_vec[index_batch_sample])
              if torch.mean(diff.view(-1)) < phaseless_loss:
                  phaseless_loss_all = diff 
                  phaseless_loss = torch.mean(diff.view(-1))
                  phase_ordering[index_batch_sample]=int(index_phase_vector)

          batch_loss[index_batch_sample] = phaseless_loss_all
      return batch_loss, phase_ordering

def combined_loss(phase_vector_nacs, n_states, n_socs, all_states, device):
    phase_vector_nacs_1 = torch.Tensor(phase_vector_nacs.size()[0],all_states).to(device)
    phase_vector_nacs_1[:,:n_states['n_singlets']+n_states['n_triplets']] = phase_vector_nacs
    #append the triplet part
    phase_vector_nacs_1[:,n_states['n_singlets']+n_states['n_triplets']:n_states['n_singlets']+n_states['n_triplets']*2] = phase_vector_nacs[:,n_states['n_singlets']:]
    phase_vector_nacs_1[:,n_states['n_singlets']+n_states['n_triplets']*2:n_states['n_singlets']+n_states['n_triplets']*3] = phase_vector_nacs[:,n_states['n_singlets']:]

    phase_vector_nacs_2 = torch.Tensor(phase_vector_nacs.size()[0],all_states).to(device)
    phase_vector_nacs_2[:,:n_states['n_singlets']] = phase_vector_nacs_1[:,:n_states['n_singlets']]
    phase_vector_nacs_2[:,n_states['n_singlets']:] = phase_vector_nacs_1[:,n_states['n_singlets']:] * -1

    #two possibilities - the min function should be give the correct error
    #therefore, build a matrix that contains all the two possibilities of phases by building the outer product of each phase vector for a given sample of a mini batch
    complex_diagonal_phase_matrix_1 = torch.Tensor(phase_vector_nacs.size()[0],n_socs).to(device)
    phase_matrix_1 = torch.Tensor(phase_vector_nacs.size()[0],all_states,all_states).to(device)
    complex_diagonal_phase_matrix_2 = torch.Tensor(phase_vector_nacs.size()[0],n_socs).to(device)
    phase_matrix_2 = torch.Tensor(phase_vector_nacs.size()[0],all_states,all_states).to(device)
   #build the phase matrix
    for sample_in_minibatch in range(phase_vector_nacs.size()[0]):
        phase_matrix_1[sample_in_minibatch] = torch.ger(phase_vector_nacs_1[sample_in_minibatch],phase_vector_nacs_1[sample_in_minibatch])
        phase_matrix_2[sample_in_minibatch] = torch.ger(phase_vector_nacs_2[sample_in_minibatch],phase_vector_nacs_2[sample_in_minibatch])
    diagonal_phase_matrix_1=phase_matrix_1[:,torch.triu(torch.ones(all_states,all_states)) == 0]
    diagonal_phase_matrix_2=phase_matrix_2[:,torch.triu(torch.ones(all_states,all_states)) == 0]
    for i in range(int(n_socs/2)):
      complex_diagonal_phase_matrix_1[:,2*i] = diagonal_phase_matrix_1[:,i]
      complex_diagonal_phase_matrix_1[:,2*i+1] = diagonal_phase_matrix_1[:,i]
      complex_diagonal_phase_matrix_2[:,2*i] = diagonal_phase_matrix_2[:,i]
      complex_diagonal_phase_matrix_2[:,2*i+1] = diagonal_phase_matrix_2[:,i]


    return complex_diagonal_phase_matrix_1, complex_diagonal_phase_matrix_2, phase_matrix_1[:,torch.triu(torch.ones(all_states,all_states)) == 1], phase_matrix_2[:,torch.triu(torch.ones(all_states,all_states)) == 1]


def delta_e_loss(target, predicted):
    delta_target = target[..., :, None] - target[..., None, :]
    delta_predicted = predicted[..., :, None] - predicted[..., None, :]

    diff_delta = (delta_target - delta_predicted) ** 2
    loss = torch.mean(diff_delta.view(-1))

    return loss


class GlobalRepresentation(nn.Module):
    """
    Utility module for generating global reps from atomwise input.
    TODO: Check whether activation function makes sense here.
    """

    def __init__(self, n_features, n_out=None, n_internal=None, aggregation_mode='sum',
                 activation=spk.nn.activations.shifted_softplus):
        super(GlobalRepresentation, self).__init__()

        if n_out is None:
            n_out = n_features

        if n_internal is None:
            n_internal = n_features

        self.transform_representation = spk.nn.Dense(n_features, n_internal, activation=activation)
        self.aggregator = spk.nn.Aggregate(1, aggregation_mode == 'mean', keepdim=False)
        self.transform_aggregate = spk.nn.Dense(n_internal, n_out, activation=activation)

    def forward(self, representation, mask=None):
        global_representation = self.transform_representation(representation)
        global_representation = self.aggregator(global_representation, mask=mask)
        global_representation = self.transform_aggregate(global_representation)
        return global_representation


class SocsTransform(nn.Module):

    def __init__(self, nsocs, real=True):
        super(SocsTransform, self).__init__()

        if real:
            soc_mask = torch.zeros(nsocs)
            soc_mask[::2] = 1
            self.register_buffer('soc_mask', soc_mask[None, :])
        else:
            self.soc_mask = None

    def forward(self, socs):

        if self.soc_mask is not None:
            return socs * self.soc_mask
        else:
            return self._transform_socs(socs)

    def _transform_socs(self, socs):
        # TODO: This should do something meaningful in the future
        return socs


class DipoleMask(nn.Module):

    def __init__(self, n_states, n_singlets=1, n_triplets=None, expansion=3, fit=False, true_inverse=True):
        super(DipoleMask, self).__init__()
        self.n_singlets=n_singlets
        self.n_triplets=n_triplets
        self.nmstates=n_triplets*3+n_singlets
    def forward(self, dipole_moments):
        # Construct the mask
        # transition dipole moments are only between singlet-singlet and triplet-triplet with same magnetic moment
        dipole_triu_mask = torch.triu(torch.ones(self.n_singlets+3*self.n_triplets,self.n_singlets+3*self.n_triplets), 0).long() == 1
        dipole_triu_mask[:self.n_singlets,self.n_singlets:]=0

        if self.n_triplets is not None:
          dipole_triu_mask[:self.n_singlets+self.n_triplets,self.n_singlets+self.n_triplets:]=0
          dipole_triu_mask[:self.n_singlets+2*self.n_triplets,self.n_singlets+self.n_triplets*2:]=0

        #take only the upper triangular of each matrix (x,y, and z) since only those values are learned 
        one_direction_mask=torch.flatten(dipole_triu_mask[torch.triu(torch.ones(self.nmstates,self.nmstates)) == 1])
        self.dipole_mask=one_direction_mask.unsqueeze(1).repeat(1,3).type(torch.float)
        device=dipole_moments.device
        self.dipole_mask = self.dipole_mask.to(device)
        return dipole_moments * self.dipole_mask

class ApproximateInverse(nn.Module):

    def __init__(self, n_states, n_triplets=None, expansion=3, fit=False, true_inverse=True):
        super(ApproximateInverse, self).__init__()

        # Construct the mask
        triu_mask = torch.triu(torch.ones(n_states, n_states), 1).long() == 1
        if n_triplets is not None:
            triu_mask[:n_triplets, n_triplets:] = 0

        self.fit = fit
        self.true_inverse = true_inverse

        if self.fit:
            alpha = torch.randn(expansion) * 10
            beta = torch.randn(expansion)
            self.alpha = nn.Parameter(alpha)
            self.beta = nn.Parameter(beta)
        else:
            # Hardcoded parameters obtained via fit
            # Laplacian
            a = torch.FloatTensor([1.13958937, 0.45577329, 2.13002577])
            b = torch.FloatTensor([1.3428374, 0.74349656, 2.31989616])
            # Gaussian
            # a = torch.FloatTensor([1.099793, 0.229364, 3.649236])
            # b = torch.FloatTensor([1.266626, 0.763655, 2.154350])
            self.register_buffer('alpha', a)
            self.register_buffer('beta', b)

        self.register_buffer('triu_mask', triu_mask)

    def forward(self, energies):
        delta_energies = energies[:, :, None] - energies[:, None, :]
        delta_energies = torch.abs(delta_energies[:, self.triu_mask])
        if self.true_inverse:
            # For testing true inverse
            approximate_inverse = 1.0 / (delta_energies + 1.0e-6)
        else:
            # Compute Laplacian approximation to inverse
            exponent = -(self.alpha ** 2)[None, None, :] * delta_energies[:, :, None]  # ** 2
            exponential = (self.beta ** 2)[None, None, :] * torch.exp(exponent)
            approximate_inverse = torch.sum(exponential, 2)

        return approximate_inverse
