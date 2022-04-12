from sys import argv, exit
import os
import numpy as np
import mdtraj as md
import MDAnalysis as mda
import pyemma
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
    eigenvalues = dict()
    eigenvectors = dict()
    times = [i ** 6 for i in range(3,11)]
    output_file = "{}/QH_cart_entropies.out".format(home_dir)
    if os.path.exists(output_file):
        os.remove(output_file)

    for res in ["ETG", "E1G", "E2G", "E3G"]:

        # Store the QH entropy values from individual runs
        qh_entropies = []
        convergences[res] = np.zeros((4,len(times)))
        eigenvalues[res] = np.zeros((4,75))
        eigenvectors[res] = np.zeros((4,75,75))

        for run in range(0,4):
            # Go to the directory containing trajectory data
            os.chdir(home_dir)
            os.chdir("{0}/run_{1}".format(res,run))

            # File names for the trajectory and topology
            traj_file = "{0}_1us_run{1}_fitted.xtc".format(res,run)
            top = "../run_0/{0}_1us_run0.gro".format(res)

            # Load in trajectory
            if False:#os.path.exists("traj_cart.npy"):
                traj = np.load("traj_cart.npy")
            else:
                traj = pyemma.coordinates.load(traj_file, top=top, features=None)
                np.save("traj_cart.npy", traj)

            if run == 0:
                combined_traj = traj
            else:
                combined_traj = np.concatenate((combined_traj,traj), axis=0)

            # Get the QH entropy for the run, and store in a list
            qh_entropy, eigenvals, eigenvecs = get_qh_entropy(traj, top, output_file, path="{0}/{1}".format(home_dir,res))
            qh_entropies.append(qh_entropy)
            eigenvalues[res][run,:] = eigenvals
            eigenvectors[res][run,:,:] = eigenvecs

            convergences[res][run, :] = plot_convergence_single(traj, top, \
                                     times, home_dir, output_file)

        np.save("{0}/{1}/combined_traj.npy".format(home_dir,res), combined_traj)
        # Get the average and standard deviation from four runs and store as a
        # tuple in a dictionary
        entropy_ave = np.average(qh_entropies)
        entropy_std = np.std(qh_entropies)
        res_qh_entropies[res] = (entropy_ave, entropy_std)
        plot_convergence_combined(convergences, times, home_dir)

    plot_eigenvals(eigenvalues, home_dir)
    plot_eigenvals_eigenvecs(eigenvalues, eigenvectors, home_dir)

    # Print the entropy estimate for each residue
    with open(output_file, "a") as file:
        for key, item in res_qh_entropies.items():
            file.write("For the residue {0}, the QH estimate of entropy is {1}"\
                       "(+/-) {2}\n".format(key,np.round(item[0],1),\
                       np.round(item[1],1)))

    relative_entropy(res_qh_entropies, output_file)

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
    plt.savefig("{}/QH_cart_convergence.png".format(home_dir))

    return None

def plot_eigenvals(eigenvalues, home_dir):

    fig, ax = plt.subplots(figsize=(16,12), constrained_layout=True)

    # Variables for plot aesthetics
    colors = {"ETG" : "dodgerblue", "E1G" : "darkorange", "E2G" : "deeppink", \
              "E3G" : "blueviolet"}
    font = {'color': 'black', 'weight': 'semibold', 'size': 28}
    print("The first EVs of E2G:", eigenvalues["E2G"][:,0])
    print("The first EVs of E3G:", eigenvalues["E3G"][:,0])

    # Make subplots for each resiude
    for res in eigenvalues.keys():

        i_vals = range(10)

        # Get average and std data for the residue
        averages = [np.average(eigenvalues[res][:,i]) for i in i_vals]
        errs = [np.std(eigenvalues[res][:,i]) for i in i_vals]

        # Make a plot with errorbars
        ax.scatter(i_vals, averages[:10], label=res, linewidth=3, \
                color=colors[res])
        ax.errorbar(i_vals, averages[:10], yerr=errs[:10], linewidth=3, \
                    color=colors[res])

        # Plot labels, ticks, borders etc.
        ax.set_xlabel("Eigenvalue", fontdict=font, labelpad=5)
        #ax.set_ylabel("",""ontdict=font, labelpad=5)
        ax.tick_params(axis='x', labelsize=20, direction='in', width=2, \
                        length=5, pad=10)
        ax.tick_params(axis='y', labelsize=20, direction='in', width=2, \
                        length=5, pad=10)
        for i in ["top","bottom","left","right"]:
            ax.spines[i].set_linewidth(2)
        ax.grid(True)
        ax.legend(fontsize=26, loc=4)

        # Add text boxes
        label = "The first EV of E2G:\n{0}\n\nThe first EV of E3G:\n{1}"\
                    .format(eigenvalues["E2G"][:,0],eigenvalues["E3G"][:,0])
        ax.text(0.98, 0.98, label, color='black', fontsize=18, \
            va="top", transform=ax.transAxes, ha="right", \
            bbox=dict(facecolor='gainsboro', edgecolor='dimgrey', \
            boxstyle='round,pad=0.5', alpha=0.7))

    # Save plot to file
    plt.savefig("{}/QH_cart_eigenvalues.png".format(home_dir))

    return None

def plot_eigenvals_eigenvecs(eigenvalues, eigenvectors, home_dir):

    fig, axes = plt.subplots(2, 2, figsize=(16,12), constrained_layout=True)

    # Variables for plot aesthetics
    colors = {"ETG" : "dodgerblue", "E1G" : "darkorange", "E2G" : "deeppink", \
              "E3G" : "blueviolet"}
    font = {'color': 'black', 'weight': 'semibold', 'size': 20}

    # Make subplots for each resiude
    for res in eigenvalues.keys():

        for j, ax in enumerate(axes.flat):

            # Plot for the eigenvalues
            if ax == axes.flat[0]:

                i_vals = range(10)

                # Get average and std data for the residue
                averages = [np.average(eigenvalues[res][:,i]) for i in i_vals]
                errs = [np.std(eigenvalues[res][:,i]) for i in i_vals]

                # Make a plot with errorbars
                ax.scatter(i_vals, averages[:10], label=res, linewidth=3, \
                        color=colors[res])
                ax.errorbar(i_vals, averages[:10], yerr=errs[:10], linewidth=3, \
                            color=colors[res])
                ax.set_ylabel("Eigenvalue",fontdict=font, labelpad=5)

                # Add text boxes
                label = "The first EV of E2G:\n{0}\n\nThe first EV of E3G:\n{1}"\
                            .format(eigenvalues["E2G"][:,0],eigenvalues["E3G"][:,0])
                ax.text(0.97, 0.95, label, color='black', fontsize=12, \
                    va="top", transform=ax.transAxes, ha="right", \
                    bbox=dict(facecolor='gainsboro', edgecolor='dimgrey', \
                    boxstyle='round,pad=0.5', alpha=0.7))

            else:

                i_vals = range(75)

                # Get average and std data for the residue
                averages = [np.average(eigenvectors[res][:,j,i]) for i in i_vals]
                errs = [np.std(eigenvectors[res][:,j,i]) for i in i_vals]

                # Make a plot with errorbars
                ax.scatter(i_vals, averages, label=res, linewidth=1, \
                        color=colors[res])
                ax.errorbar(i_vals, averages, yerr=errs, linewidth=1, \
                            color=colors[res])

                ax.set_ylabel("Eigenvector {}".format(j),fontdict=font, labelpad=5)

            ax.tick_params(axis='x', labelsize=20, direction='in', width=2, \
                            length=5, pad=10)
            ax.tick_params(axis='y', labelsize=20, direction='in', width=2, \
                            length=5, pad=10)
            for i in ["top","bottom","left","right"]:
                ax.spines[i].set_linewidth(2)
            ax.grid(True)
            ax.legend(fontsize=16, loc=4)

    # Save plot to file
    plt.savefig("{}/QH_cart_EVs.png".format(home_dir))

    return None

def plot_convergence_single(traj, top, times, home_dir, output_file):
    """Check the convergence of entropy as a function of time.

    Parameters
    ----------
    traj : ndarray
        Trajectory loaded from an .xtc file.
    top : str
        The .gro file name for loading a topology.
    times : int list
        Simulation time segments for which the entropy should be calculated.
    home_dir : str
        The path to the main working directory.
    output_file : str
        The path to the output file in the main working directory.
    """
    # Get entropies as a function of simulation time
    entropies = []

    for t in times:
        entropy, _, _ = get_qh_entropy(traj, top, output_file, time=t)
        entropies.append(entropy)

    # System description and font dictionary for plots
    # residue = top.split("_")[0]
    # font = {'color': 'black', 'weight': 'semibold', 'size': 18}
    #
    # # Make a figure of the convergence
    # fig, ax = plt.subplots(constrained_layout=True)
    # ax.plot(np.divide(times, 1000), entropies)
    # ax.scatter(np.divide(times, 1000), entropies)
    # ax.set_xlabel("Time (ns)", fontdict=font, labelpad=5)
    # ax.set_ylabel(r"QH Entropy (J / mol$\cdot$ K)", fontdict=font, labelpad=5)
    # ax.tick_params(axis='y', labelsize=14, direction='in', width=2, \
    #                 length=5, pad=10)
    # ax.tick_params(axis='x', labelsize=14, direction='in', width=2, \
    #                 length=5, pad=10)
    # for i in ["top","bottom","left","right"]:
    #     ax.spines[i].set_linewidth(2)
    # ax.grid(True)
    #
    # plt.savefig("{0}/{1}/convergence.png".format(home_dir, residue))

    return entropies

def get_qh_entropy(traj, top, output_file, path=None, time=None):
    """Get the (cartesian) QH entropy for a specific trajectory.

    The working directory should be the directory where the files trajectory and
    topology files are contained.

    Parameters
    ----------
    traj : ndarray
        Trajectory loaded from an .xtc file.
    top : str
        The .gro file name for loading a topology.
    output_file : str
        The path to the output file in the main working directory.
    time : int
        Time of the trajectory in picoseconds which should be considered for the
        analysis. Used when determining the convergence as a function of
        simulation time.
    Returns
    -------
    entropy : float
        The QH entropy estimate.
    """
    # System description
    residue = top.split("_")[0]
    run = top.split("run")[1][0]

    # read in relevant lines from the topology file, add to dictiory "atoms"
    topol = read_topol()
    atoms = dict()
    for atom in topol:
        atoms[atom[0]] = Atom(atom)

    # Get the diagonalized mass matrix with dim (3N x 3N)
    mass_matrix = diagonalize_mass(atoms)
    if path != None:
        np.save("{}/mass_matrix.npy".format(path), mass_matrix)

    # Determine angular frequencies from the (mass-weighted) covariance matrix
    # of the trajectory using principle component analysis
    if time != None:
        omegas, eigenvals, eigenvecs, covar = pca(traj[:time,:], mass_matrix, path)
    else:
        omegas, eigenvals, eigenvecs, covar = pca(traj, mass_matrix, path)

    # Calculate the QH entropy
    entropy = entropy_qh(omegas) * 1000 # J / mol K

    # Save estimate to file
    if time == None:
        with open(output_file, "a") as f:
            if residue == "ETG" and run == "0":
                f.write("The QH entropy estimation for 4 runs of each residue,"\
                        " using a cartesian coordinate system:\n")
            f.write("Residue: {0}, Run: {1}, Entropy: {2}\n".format(residue, \
                    run, np.round(entropy,1)))

    return entropy, eigenvals, eigenvecs

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
            exit(1)

    topol = []
    for line in top_file:
        if ";" == line[0]:
            continue
        line = line.split()
        if any(map(lambda v: v in residues, line)):
            topol.append(line)

    return topol

def diagonalize_mass(atoms):
    """Make a diagonalized mass matrix with dimensions 3N.

    Parameters
    ----------
    atoms : Atom dictionary
        Dictionary with topology information for each atom.
    Returns
    -------
    mass_matrix : ndarray
        A diagonal matrix of size 3N x 3N, where the diagonal elements are
        atomic masses, in multiples of 3.
    """
    mass_list = []
    for num, atom in atoms.items():
        mass_list.append(atom.mass)
    mass_list = [mass for mass in mass_list for i in range(3)]
    mass_matrix = np.diag(mass_list)

    return mass_matrix

def pca(traj, mass_matrix, path):
    """Extracts the angular frequencies from a trajectory to estimate entropy.

    This approach is based on a fitted structure trajectory in cartesian
    coordinates. It estimates the entropy within the quasi-harmonic
    approximation using a mass-weighted covariance matrix of the trajectory.
    The general eigenvalue problem is then solved and the eigenvalues correspond
    to angular frequencies. Only the first 3N - 6 angular frequencies are used
    to calculate the entropy since 6 degrees of freedom arrise from translation
    and rotation.

    Parameters
    ----------
    traj : ndarray
        A fitted molecular dynamics trajectory.
    mass_matrix : ndarray
        A diagonal matrix of size 3N x 3N, where the diagonal elements are
        atomic masses, in multiples of 3.
    Returns
    -------
    omegas : float list
        A list of angular frequencies with length 3N - 6.
    """
    np_val_file = "{}/eigenvals_QH_cart.npy".format(path)
    np_vec_file = "{}/eigenvecs_QH_cart.npy".format(path)
    
    # Get the covariance matrix from the zero-fitted trajectory
    mean_centered = traj - np.mean(traj, axis=0)
    covar = np.cov(mean_centered.T, bias=True)

    # Multiply the covariance matrix of the fitted structure by the mass matrix
    mass_root = np.sqrt(mass_matrix)
    mass_weighted = mass_root @ covar @ mass_root

    # Solve the eigenvalue problem
    eigenvals, eigenvecs = np.linalg.eig(mass_weighted)
    if path != None:
        np.save(np_val_file,eigenvals)
        np.save(np_vec_file,eigenvecs)

    # Extract the angular frequencies from the eigenvalues
    # Only include the first 3N - 6 eigenvalues
    omegas = [get_angular_freq(eigenval) for eigenval in eigenvals[:-6]]

    return omegas, eigenvals, eigenvecs, covar

def get_angular_freq(eigenval):
    """Calculates angular frequency from an eigenvalue of the covariance matrix.

    Parameters
    ----------
    eigenvalue : float
        A single eigenvalue from the covariance matrix.
    Returns
    -------
    omega : float
        A singular angular frequency, used in the quasi-harmonic approach to
        estimating entropy.
    """
    kb = 8.3144621 * 10 ** (-3) # kJ / mol / K
    temp = 298 # K

    omega = np.sqrt(kb * temp / eigenval)

    return omega

def entropy_qh(omegas):
    """Determine the quasi-harmonic estimate for entropy.

    Uses the angular frequencies calculated from the covariance matrix to
    determine the QH entropy using the quantum mechanical formulation for a
    harmonic oscillator. Units for constants are given in GROMACS units.

    Parameters
    ----------
    omegas : float list
        The angular frequencies determined by principle component analysis.
    Returns
    -------
    entropy : float
        A quasi-harmonic estimation of entropy in kJ / mol K.
    """
    hbar = 0.0635077993 # kJ / mol ps
    kb = 8.3144621 * 10 ** (-3) # kJ / mol / K
    temp = 298 # K
    entropy = 0 # kJ / mol / K

    for omega in omegas:
        boltz = hbar * omega /(kb * temp)
        term1 = boltz / (np.exp(boltz) - 1)
        term2 = np.log(1 - np.exp(-boltz))
        entropy += term1 - term2

    return kb * entropy

def relative_entropy(absolute_entropy, output_file):
    """Calculates relative entropy from absolute entropy.

    The calculate entropy differences are added to the file
    "qh_entropies.txt".

    Parameters
    ----------
    absolute_entropy : (float float) dict
         A dictionary of tuples, accessed with the residue as a key
         (e.g. "ETG"), containing the average absolute entropy and
         the std.
    output_file : str
        The path to the output file in the main working directory.
    """
    absolute_etg = absolute_entropy["ETG"][0]
    error_etg = absolute_entropy["ETG"][1]
    relative_s = [(s[0] - absolute_etg, s[1] + error_etg, res) \
            for res, s in absolute_entropy.items() if res != "ETG"]
    with open(output_file, "a") as file:
        file.write("\nRelative entropy values wrt ETG\n")
        for delta, err, res in relative_s:
            file.write("Delta S is {0} (+/-) {1} for {2}\n".format(np.round(\
                        delta,1), np.round(err,1), res))

if __name__ ==  '__main__':
    main(argv)
    exit(0)
