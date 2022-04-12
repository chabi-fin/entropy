from sys import argv, exit
import os
import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt

def main(argv):
    """Get the quasi-harmonic entropies of four noncanonical residues.

    Estimates the absolute and relative entropies from 4 noncanonical amino
    acids to the file "qh_entropies.txt".
    """
    home_dir = os.getcwd()

    # The key for this dictionary is the residue name (e.g. "ETG") and the item
    # stored is a (float float) tuple, where the first value is the average
    # QH entropy and the second value is the std over 4 separate runs.
    res_qh_entropies = dict()
    convergences = dict()
    eigenvectors = dict()
    covar_dets = dict()
    times = np.logspace(2,6, num=20)
    times = [i ** 6 for i in range(3,11)]
    output_file = "{}/QH_BAT_entropies.out".format(home_dir)
    if os.path.exists(output_file):
        os.remove(output_file)

    for res in ["ETG", "E1G", "E2G", "E3G"]:

        # Store the QH entropy values from individual runs
        internal_entropies = []
        convergences[res] = np.zeros((4,len(times)))
        covar_determinants = []

        for run in range(4):
            # Go to the directory containing trajectory data
            os.chdir(home_dir)
            os.chdir("{0}/run_{1}".format(res,run))

            # File names for the trajectory and topology
            traj_file = "{0}_1us_run{1}_fitted.xtc".format(res,run)
            top = "{0}_1us_run{1}.pdb".format(res,run)

            # Load in trajectory
            if False:# os.path.exists("traj_internal.npy"):

                # The loaded traj has 3N DOF
                trajBAT = np.load("traj_internal.npy")

            else:
                traj = mda.Universe(top, traj_file)
                print(traj)
                print(traj.residues)
                selected_residues = traj.select_atoms("resid 1-3")

                trajBAT, inds = get_BAT_traj(selected_residues)
                np.save("traj_internal.npy", trajBAT)

            #print("SHAPE of combined traj:",combined_traj.shape)
            if run == 0:
                combined_traj = trajBAT
            else:
                combined_traj = np.concatenate((combined_traj,trajBAT), axis=0)

            # Calculate the absolute entropy for the run
            absolute_S, covar_det = get_absolute_S(trajBAT)
            print(absolute_S)
            internal_entropies.append(absolute_S)
            covar_determinants.append(covar_det)

            convergences[res][run, :] = get_convergence(trajBAT, times)
            #jacobians, ave_jacobian = get_jacobian_det(trajBAT)
            #print("Jacobian:", ave_jacobian)

        # Get the average and standard deviation from four runs and store as a
        # tuple in a dictionary
        entropy_ave = np.average(internal_entropies)
        entropy_std = np.std(internal_entropies)
        covar_dets[res] = np.average(covar_determinants)
        res_qh_entropies[res] = (entropy_ave, entropy_std)
        plot_convergence_combined(convergences, times, home_dir)
        combined_entropy = get_absolute_S(combined_traj)
        print("Combined entropy is", combined_entropy)

    # Print the entropy estimate for each residue
    with open(output_file, "w") as file:
        for key, item in res_qh_entropies.items():
            file.write("For the residue {0}, the QH estimate of entropy is {1}"\
                       "(+/-) {2}\n".format(key,np.round(item[0],1),\
                       np.round(item[1],1)))

    relative_entropy(res_qh_entropies, covar_dets, output_file)

    return None

class Atom():
    """A class to represent individual atoms in the topology.

    Attributes
    ----------
    num : int
        The number of the atom in the ordered topology, indexed from one.
    type : str
        The atom type, which depends on the force field, e.g. "CX", "HP" etc.
    residue : str
        The 3 or 4 letter residue code, e.g. "TYR", "NTYR" etc.
    name : str
        The atom name, unique within the residue, e.g. "CG", "HD1" etc.
    mass : float
        The atomic mass, e.g. 12.01, 1.008 etc.
    Methods
    -------
    str():
        Prints the attributes of the atom.
    """
    def __init__(self, topol):
        """Constructs the attributes of an atom from the topology.

        Parameters
        ----------
        topol : str list
            The corresponding line in the topology, split by whitespace.
        """
        self.num = topol[0]
        self.type = topol[1]
        self.residue = topol[3]
        self.name = topol[4]
        self.mass = float(topol[7])

    def __str__(self):
        """Prints the attributes of the Atom."""
        return "Number: {}, Type: {}, Residue: {}, Name: {}, Mass: {}00"\
            .format(self.num, self.type, self.residue, self.name, self.mass)

def read_topol():
    """Read in a topology file.

    Extracts topology information related to the individual atoms and bonds.
    This function will only work on residues listed in the "residues" list, so
    add more residues as needed.
    Parameters
    ----------
    topol : file
        Topology file to read in, where the default is identical to the GROMACS
        default, "topol.top".
    Returns
    -------
    atoms : (str list) list
        Each atom corresponds to a line from the topology file, which is split
        by whitespace.
    bonds : (str list) list
        List of directly bonded atom pairs.
    """
    residues = ["ACE","NME","ETG","E1G","E2G","E3G"]
    try:
        with open("topol.top", "r") as file:
            top_file = file.readlines()
    except OSError:
            print("topology file not found.")
            os.sys.exit(1)

    topol = []
    for line in top_file:
        if ";" == line[0]:
            continue
        line = line.split()
        if any(map(lambda v: v in residues, line)):
            topol.append(line)

    return topol

def get_BAT_traj(traj):
    """Convert a trajectory to BAT coordinates and extract internal indicies.

    The 3N - 6 internal coordinate trajecory is returned, along with the
    indicies for atoms involved in each coordinate. Each coordinate may contain
    2, 3 or 4 indicies for bonds, angles and torsions, respectively.

    Parameters
    ----------
    traj : MDAnalysis.core.universe.Universe
        The trajectory in cartesian coordinates.
    Returns
    -------
    trajBAT : ndarray
        The trajectory as an ndarray in BAT coordinates.
    inds : (int list) list
        Each entry in the list will have 2, 3 or 4 indicies, corresponding to an
        internal DOF.
    """
    # Use mda package to convert the trajectory from CART to BAT
    from MDAnalysis.analysis.bat import BAT
    bat = BAT(traj)
    bat.run()
    bat.save("test.npy")
    # Ignore the initial coordinates which correspond to external DOFs
    trajBAT = bat.results.bat
    np.save("trajBAT.npy",trajBAT)

    # The indicies for atoms involved in each internal coordinate
    dof = np.size(trajBAT, axis=1)
    inds = [None] * dof

    # The first 3 coordinates correspond to root atoms, with 2 bonds and 1 angle
    root = bat._root
    inds[0] = ([root[0].ix, root[1].ix])
    inds[1] = ([root[1].ix, root[2].ix])
    inds[2] = ([root[0].ix, root[1].ix, root[2].ix])

    # Add indicies for bond, angle and torsion coordinates
    torsions = bat._torsions
    num_tors = len(torsions)
    for i, tor in enumerate(bat._torsions):
        inds[3+i] = [tor[0].ix, tor[1].ix]
        inds[num_tors+3+i] = [tor[0].ix, tor[1].ix, tor[2].ix]
        inds[2*num_tors+3+i] = [tor[0].ix, tor[1].ix, tor[2].ix, tor[3].ix]

    return trajBAT, inds

def get_absolute_S(trajBAT, time=None):
    """Determines the absolute entropy for internal coordinates.

    Parameters
    ----------
    trajBAT : ndarray
        The trajectory as an ndarray in BAT coordinates
    Returns
    -------
    entropy : float
        The absolute entropy for the run.
    covar_det : float
        The determinant of the covariance matrix
    """
    # Get the covariance matrix from the fitted trajectory
    zeroT = trajBAT[:,6:] - np.mean(trajBAT[:,6:], axis=0)
    if time != None:
        covar = np.cov(zeroT[:time,:].T)
    else:
        covar = np.cov(zeroT.T)

    # Find the determinant of the covariance matrix
    #if False:#os.path.exists("covar_det_BAT.npy"):
    #    covar_det = np.load("covar_det_BAT.npy")
    #else:
    #    covar_det = np.linalg.det(covar)

    # Get the eigenvectors of the covar matrix and save to file
    eigenvals, eigenvecs = np.linalg.eig(covar)
    lncovar_det = 0
    for eigenval in eigenvals:
        lncovar_det += np.log(eigenval)
    covar_det = np.exp(lncovar_det)
    np.save("eigenvecs_QH_BAT.npy",eigenvecs)

    # Estimate using the Karplus Kushick approach
    gas_const = 8.314462618 # J / mol K
    dof = np.size(trajBAT, axis=1) - 6
    jacobian_det = get_jacobian_det(trajBAT)
    external_contribution = gas_const * np.log(8 * np.pi ** 2 * jacobian_det)
    entropy = gas_const / 2 * dof + gas_const / 2 * np.log((2*np.pi) ** dof * covar_det)
    return entropy, covar_det

def relative_entropy(absolute_entropy, covar_dets, output_file):
    """Calculates relative entropy from absolute entropy.

    The calculate entropy differences are added to the file "qh_entropies.out".

    Parameters
    ----------
    absolute_entropy : (float float) dict
         A dictionary of tuples, accessed with the residue as a key
         (e.g. "ETG"), containing the average absolute entropy and the std.
    covar_dets : float dict
        Dictionary of the average covariance matrix determinant for each
        residue.
    output_file: str
        The path to the output_file in the main working directory.
    """
    kb = 8.314462618 # J / mol K
    absolute_etg = absolute_entropy["ETG"][0]
    error_etg = absolute_entropy["ETG"][1]
    relative_s = [(s[0] - absolute_etg, s[1] + error_etg, res) for res, s \
                    in absolute_entropy.items() if res != "ETG"]
    relative_s_covar_det = [(kb/2 * np.log(s / covar_dets["ETG"]), res) \
                    for res, s in covar_dets.items() if res != "ETG"]
    with open(output_file, "a") as file:
        file.write("\nRelative entropy values wrt ETG\n")
        for delta, err, res in relative_s:
            file.write("Delta S is {0} (+/-) {1} for {2}\n".format(np.round(\
                        delta,1), np.round(err,1), res))
        file.write("\nRelative entropy values wrt ETG using the covariance "\
                    "determinant.\n")
        for delta, res in relative_s_covar_det:
            file.write("Delta S is {0} for {1}\n".format(np.round(\
                        delta,1), res))

    return None

def get_convergence(trajBAT, times):
    """Check the convergence of entropy as a function of time.

    Parameters
    ----------
    traj : ndarray
        The trajectory as an ndarray in BAT coordinates
    times : int list
        Simulation time segments for which the entropy should be calculated.
    Returns
    -------
    entropies : float list
        Relative entropies calculated for different simulation times
        (J / mol K).
    """
    # Get entropies as a function of simulation time
    entropies = []

    # Determine for each given time
    for t in times:
        entropy, _ = get_absolute_S(trajBAT, time=t)
        entropies.append(entropy)

    return entropies

def plot_convergence_combined(convergences, times, home_dir):
    """Check the convergence of entropy as a function of time.

    Parameters
    ----------
    convergences : ndarray dict
        The entropies are stored as arrays and accessed using the residue (e.g.
        "ETG") as a key. The array columns are for the different runs and the
        rows store entropies for different simulation times, according to the
        values in "times".
    times : int list
        Simulation time segments for which the entropy should be calculated.
    home_dir : str
        The path to the main working directory.
    """
    # Make a figure of the convergence
    fig, ax = plt.subplots(figsize=(16,12), constrained_layout=True)

    # Variables for plot aesthetics
    colors = {"ETG" : "dodgerblue", "E1G" : "darkorange", "E2G" : "deeppink", \
              "E3G" : "blueviolet"}
    font = {'color': 'black', 'weight': 'semibold', 'size': 28}

    # Make subplots for each resiude
    for res in convergences.keys():

        # Get average and std data for the residue
        averages = [np.average(convergences[res][:,time]) \
                    for time in range(len(times))]
        errs = [np.std(convergences[res][:,time]) for time in range(len(times))]

        # Make a plot with errorbars
        ax.plot(np.divide(times, 1000), averages, label=res, linewidth=3, \
                color=colors[res])
        ax.errorbar(np.divide(times, 1000), averages, yerr=errs, linewidth=3, \
                    color=colors[res])

        # Plot labels, ticks, borders etc.
        if res == "E2G" or "E3G":
            ax.set_xlabel("Time (ns)", fontdict=font, labelpad=5)
        ax.set_ylabel(r"Conformational Entropy (J / mol$\cdot$ K)", \
                          fontdict=font, labelpad=5)
        ax.tick_params(axis='x', labelsize=20, direction='in', width=2, \
                        length=5, pad=10)
        ax.tick_params(axis='y', labelsize=20, direction='in', width=2, \
                        length=5, pad=10)
        for i in ["top","bottom","left","right"]:
            ax.spines[i].set_linewidth(2)
        ax.grid(True)
        ax.legend(fontsize=26, loc=4)

    # Save plot to file
    plt.savefig("{}/QH_BAT_convergence_Karplus.png".format(home_dir))

    return None

def get_jacobian_det(trajBAT):
    """Find the determinant of the jacobian matrix as a function of time.

    Parameters
    ----------
    trajBAT: ndarray
        The trajectory in internal coordinates.
    Returns
    -------
    jacobian : float
        The jacobian determinant used in converting mass weighting to internal
        coordinates.
    """
    # Get the average bonds and average angles into lists
    average_coords = list(np.mean(trajBAT, axis=0))
    n_atoms = int(len(average_coords) / 3)
    bond_coords = average_coords[6:8] + average_coords[9:n_atoms+6]
    angle_coords = [average_coords[8]] + average_coords[n_atoms+6:2*n_atoms+3]

    # Calculate the Jacobian using average coordinates
    jacobian_det = np.sin(average_coords[3]) * (bond_coords[0] ** 2)
    for bond, angle in zip(bond_coords[1:],angle_coords):
        jacobian_det *= np.sin(angle) * (bond ** 2)

    return jacobian_det

if __name__ ==  '__main__':
    main(argv)
    exit(0)
