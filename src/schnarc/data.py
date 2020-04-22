class Properties:
    """
    Common properties
    """
    energy = 'energy'
    forces = 'forces'
    socs = 'socs'
    nacs = 'nacs'
    dipole_moment = 'dipoles'
    charges = 'charges'

    # Only for prediction and calculator
    hessian = 'hessian'

    # Available properties
    properties = [
        energy,
        forces,
        socs,
        nacs,
        dipole_moment
    ]

    # Properties for which normalization is meaningful
    normalize = [
        energy,
        socs
    ]

    # Properties with potential phase issues
    phase_properties = [
        dipole_moment,
        socs,
        nacs
    ]

    # Hessians available
    hessians_available = [
        energy
    ]

    # Standard mappings for properties
    mappings = {
        energy: (energy, 'y'),
        forces: (energy, 'dydx'),
        socs: (socs, 'y'),
        dipole_moment: (dipole_moment, 'y'),
        charges: (dipole_moment, 'yi'),
        nacs: (nacs, 'dydx')
    }

    n_triplets = 'n_triplets'
    n_singlets = 'n_singlets'
    n_states = 'n_states'
