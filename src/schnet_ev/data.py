class Charges:
     
    hirshfeld_pbe = "hirshfeld_pbe"
    hirshfeld_pbe0 = "hirshfeld_pbe0"
    hirshfeld_pbe0_h2o = "hirshfeld_pbe0_h2o"
    delta_hirshfeld = "delta_hirshfeld_pbe0_h2o"
 
    properties = [
        hirshfeld_pbe,
        hirshfeld_pbe0,
        hirshfeld_pbe0_h2o, 
        delta_hirshfeld
    ]
class Delta_EV_properties:

    delta_eigenvalues_pbe0_h2o = "delta_eigenvalues_pbe0_h2o"
    delta_eigenvalues_pbe0_gbw = "delta_eigenvalues_pbe0_gbw"
    delta_eigenvalues_pbe_gbw = "delta_eigenvalues_pbe_gbw"
    delta_eigenvalues_pbe_qzvp = "delta_eigenvalues_pbe_qzvp"
    delta_eigenvalues_pbe0_qzvp = "delta_eigenvalues_pbe0_qzvp"

    properties = [
        delta_eigenvalues_pbe0_h2o,
        delta_eigenvalues_pbe_gbw,
        delta_eigenvalues_pbe0_gbw,
        delta_eigenvalues_pbe_qzvp,
        delta_eigenvalues_pbe0_qzvp
    ]
 
class Eigenvalue_properties:

    eigenvalues_active = "eigenvalues_active"
    eigenvalues_pbe = "eigenvalues_pbe"
    eigenvalues_pbe0 = "eigenvalues_pbe0"
    eigenvalues_pbe0_h2o = "eigenvalues_pbe0_h2o"
    eigenvalues_pbe0_qzvp = "eigenvalues_pbe0_qzvp"
    eigenvalues_gbw = "eigenvalues_gbw"
    eigenvalues = 'eigenvalues'
    sum_occ_orb = 'sum_occ_orb'
    occ_orb = 'occ_orb'
    unocc_orb = 'unocc_orb'
    occ_orb_forces='occ_orb_forces'
    eigenvalues_forces = 'eigenvalues_forces'
    # Only for prediction and calculator
    homo_lumo = "homo_lumo"
    hessian = 'hessian'
    eigenvalues_active_forces = "eigenvalues_active_forces"

    # Available properties
    properties = [
        eigenvalues,
        occ_orb,
        unocc_orb,
        homo_lumo,
        eigenvalues_active,
        eigenvalues_pbe0,
        eigenvalues_pbe,
        eigenvalues_pbe0_h2o,
        eigenvalues_pbe0_qzvp,
        eigenvalues_gbw
       
    ]


class Properties:
    """
    Common properties
    """
    eigenvalues_active = "eigenvalues_active"
    eigenvalues_pbe = "eigenvalues_pbe"
    eigenvalues_pbe0 = "eigenvalues_pbe0"
    eigenvalues_pbe0_h2o = "eigenvalues_pbe0_h2o"
    eigenvalues_pbe0_qzvp = "eigenvalues_pbe0_qzvp"
    eigenvalues_gbw = "eigenvalues_gbw"
    hirshfeld_pbe = "hirshfeld_pbe"
    hirshfeld_pbe0 = "hirshfeld_pbe0"
    hirshfeld_pbe0_h2o = "hirshfeld_pbe0_h2o"
    delta_hirshfeld = "delta_hirshfeld_pbe0_h2o"
    energy = "energy_pbe0"
    energy = "energy_pbe0_h2o"
    delta_energy = "delta_energy_pbe0_h2o"
    delta_energy = "delta_energy_pbe0_qzvp"
    delta_eigenvalues_pbe0_h2o = "delta_eigenvalues_pbe0_h2o"
    delta_eigenvalues_pbe0_gbw = "delta_eigenvalues_pbe0_gbw"
    delta_eigenvalues_pbe_gbw = "delta_eigenvalues_pbe_gbw"
    delta_eigenvalues_pbe_qzvp = "delta_eigenvalues_pbe_qzvp"
    delta_eigenvalues_pbe0_qzvp = "delta_eigenvalues_pbe0_qzvp"
    energy = 'energy'
    occupation = "occupation"
    n_elec_molec = "n_elec_molec"
    n_elec_atom = "n_elec_atom"
    forces = 'forces'
    eigenvalues = 'eigenvalues'
    sum_occ_orb = 'sum_occ_orb'
    occ_orb = 'occ_orb'
    unocc_orb = 'unocc_orb'
    occ_orb_forces='occ_orb_forces'
    delta_E_forces = 'delta_E_forces'
    eigenvalues_forces = 'eigenvalues_forces'
    homo_lumo = 'homo_lumo'
    delta_E = 'delta_E'
    # Only for prediction and calculator
    hessian = 'hessian'
    eigenvalues_active_forces = "eigenvalues_active_forces"

    # Available properties
    properties = [
        energy,
        forces,
        eigenvalues,
        occ_orb,
        unocc_orb,
        homo_lumo,
        delta_E,
        delta_energy,
        eigenvalues_active,
        eigenvalues_pbe0,
        eigenvalues_pbe,
        eigenvalues_pbe0_h2o,
        eigenvalues_pbe0_qzvp,
        eigenvalues_gbw,
        delta_eigenvalues_pbe0_h2o,
        delta_eigenvalues_pbe_gbw,
        delta_eigenvalues_pbe0_gbw,
        delta_eigenvalues_pbe_qzvp,
        delta_eigenvalues_pbe0_qzvp,
        hirshfeld_pbe,
        hirshfeld_pbe0,
        hirshfeld_pbe0_h2o,
        delta_hirshfeld,
        occupation
        
    ]

    # Properties for which normalization is meaningful
    normalize = [
        energy,
        homo_lumo,
        delta_eigenvalues_pbe0_gbw,
        delta_eigenvalues_pbe0_h2o
    ]

    # Hessians available
    hessians_available = [
        energy
    ]

    # Standard mappings for properties
    mappings = {
        energy: (energy, 'y'),
        forces: (energy, 'dydx'),
        eigenvalues: (eigenvalues, 'y'),
        #eigenvalues_forces: (eigenvalues, 'dydx'),
        occ_orb: (occ_orb, 'y'),
        occ_orb_forces: (occ_orb, 'dydx'),
        unocc_orb: (unocc_orb, 'y'),
        delta_E: (delta_E, 'y'),
        delta_E_forces: (delta_E_forces,'dydx'),
        homo_lumo:(homo_lumo,'y'),
        sum_occ_orb: (sum_occ_orb,'y'),
        delta_energy:(delta_energy,'y'),
        eigenvalues_active: (eigenvalues_active, 'y'),
        eigenvalues_pbe0: (eigenvalues_pbe0,"y"),
        eigenvalues_pbe0_h2o: (eigenvalues_pbe0_h2o,"y"),
        eigenvalues_gbw: (eigenvalues_gbw,"y"),
        eigenvalues_pbe0_qzvp: (eigenvalues_pbe0_qzvp, "y")
         #eigenvalues_active_forces:(eigenvalues_active, 'dydx')
    }

    n_unocc = 'n_unocc'
    n_occ = 'n_occ'
    n_orb = 'n_orb'
