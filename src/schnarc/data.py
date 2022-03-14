class Properties:
    """
    Common properties
    """
    energy = 'energy'
    forces = 'forces'
    gradients = 'gradients'
    socs = 'socs'
    old_socs = "old_socs"
    nacs = 'nacs'
    dipole_moment = 'dipoles'
    charges = 'charges'
    phases = 'phases'
    # Only for prediction and calculator
    hessian = 'hessian'

    # Available properties
    properties = [
        energy,
        forces,
        gradients,
        socs,
        old_socs,
        nacs,
        dipole_moment,
        phases
    ]

    # Properties for which normalization is meaningful
    normalize = [
        energy] #,
        #socs
    #]

    # Properties with potential phase issues
    phase_properties = [
        dipole_moment,
        socs,
        old_socs,
        nacs,
        phases
    ]

    # Hessians available
    hessians_available = [
        energy
    ]

    # Standard mappings for properties
    mappings = {
        energy: (energy, 'y'),
        forces: (energy, 'dydx'),
        gradients: (energy, 'dydx'),
        socs: (socs, 'y'),
        old_socs: (old_socs, "y"),
        dipole_moment: (dipole_moment, 'y'),
        charges: (dipole_moment, 'yi'),
        nacs: (nacs, 'dydx'),
        phases: (phases, 'y')
    }

    n_triplets = 'n_triplets'
    n_singlets = 'n_singlets'
    n_states = 'n_states'
