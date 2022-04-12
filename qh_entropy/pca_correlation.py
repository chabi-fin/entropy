from sys import argv, exit
import os
import numpy as np
import pyemma
import matplotlib.pyplot as plt
import imageio

def main(argv):

    home_dir = "/home/lf1071fu/fluorinated_aa/qh_entropy"
    eigenvalues = dict()
    entropy_QH_QM = dict()
    entropy_QH_CL = dict()
    entropies_QH_QM = dict()
    entropy_QM_mode = dict()
    entropy_CL_mode = dict()
    anharmonic_corrs = dict()
    pairwise_corrs = dict()
    residues = ["ETG", "E1G", "E2G", "E3G"]
    times = [i ** 4 for i in range(3,31)]

    # The absolute entropies are determined for the individual residues
    for res in residues:

        # Grab trajectory and topology data from file
        path = "{0}/{1}".format(home_dir,res)
        trajs = get_combined_traj(path, res)
        mass_matrix, atoms = get_mass_mtx(path)
        print(res)

        BAT = False
        #trajs = np.load("{}/mw_trajBAT.npy".format(path))

        # Perform QH analysis: determine and diagonalize covariance matrix
        # Use the eigendecomposition to get a classical and quantum estimate
        # of the entropy.
        eigenvalues[res], eigenvecs = diag_covar(trajs, mass_matrix, path, \
                                                                        BAT=BAT)
        omegas = [get_angular_freq(eigval) for eigval in eigenvalues[res]]
        entropy_QH_QM[res] = get_QH_QM_entropy(omegas)
        print(np.round(entropy_QH_QM[res],1), "J / mol K")
        entropy_QH_CL[res], classical_limit = get_QH_CL_entropy(omegas)

        # Get quantum entropy as a function of simulation time
        entropies_QH_QM[res] = get_convergence(trajs, mass_matrix, times, \
                                                                path, BAT=BAT)

        # Determine the entropy contribution of each mode by quantum and
        # classical estimates.
        entropy_QM_mode[res], entropy_CL_mode[res] = get_entropies_by_mode(omegas)

        #if BAT:
        #    analyze_eigenvecs(eigenvecs, atoms, path)

        # Obtain the projected trajectory
        #QH_coords = project_traj(trajs, eigenvecs, np.sqrt(mass_matrix), BAT=BAT)

        # Visualize the most important mass-weighted principle components
        #num_PCs = get_num_PCs(eigenvalues[res])
        #correlation_plots(QH_coords, res, 10, path, BAT=BAT)

        # Anharmonic corrections to the entropy
        #opt_bin_width(eigenvalues[res], QH_coords, path, BAT=BAT)
        #anharmonic_corrs[res], anharm_mode = get_anharmonic_corr(QH_coords, \
        #                eigenvalues[res], entropy_CL_mode[res], \
        #                classical_limit, 298, path, BAT=BAT)

        # Pairwise supralinear correlations
        #opt_bin_width2D(eigenvalues[res], QH_coords, path, BAT=BAT)
        #pairwise_corrs[res], pairwise_mn = get_pairwise_corr(QH_coords, \
        #            eigenvalues[res], anharm_mode, classical_limit, 298, path)
        #plot_pairwise_corrs(pairwise_mn, classical_limit, path)

    # Combined plots for the comparing outcomes between the residues
    #plot_eigenvals(eigenvalues, home_dir, BAT=BAT)
    #plot_convergence_combined(entropies_QH_QM, times, home_dir, BAT=BAT)
    plot_entropies_by_mode_poster(entropy_QM_mode, entropy_CL_mode, home_dir, BAT=BAT)

    # Use the absolute entropies to determine the entropy differences.
    #summarize_results(entropy_QH_QM, anharmonic_corrs, pairwise_corrs, \
    #                                                        home_dir, BAT=BAT)

    return None

# Functions related to uncorrected estimates of the entropy
def diag_covar(traj, mass_matrix, path, BAT=False):
    """Perform mass-weighted principal component analysis.

    Parameters
    ----------
    traj : ndarray
        The trajectory in BAT coordinates.
    mass_matrix : ndarray
        The 3N x 3N diagonal mass matrix.
    path : str
        The full path to the working directory.

    Returns
    -------
    eigenvals : ndarray
        The eigenvalues of the covariance matrix.
    eigenvecs : ndarray
        The eigenvectors of the covariance matrix.
    covar : ndarray
        The mass-weighted covariance matrix of the trajectory.
    mass_root : ndarray
        The square root of the diagonalized mass matrix.

    """
    if BAT:
        mw_covar = np.cov(traj.T)
    else:
        traj_mean = np.mean(traj, axis=0)
        covar = np.cov((traj - traj_mean).T)
        mass_root = np.sqrt(mass_matrix)
        mw_covar = mass_root @ covar @ mass_root

    # Solve the eigenvalue problem; order by descending eigenvalues
    eigenvals, eigenvecs = np.linalg.eig(mw_covar)
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:,idx]

    if BAT:
        np.save("{}/eigenvals_QH_BAT.npy".format(path),eigenvals)
        np.save("{}/eigenvecs_QH_BAT.npy".format(path),eigenvecs)
    else:
        np.save("{}/eigenvals_QH_cart.npy".format(path),eigenvals)
        np.save("{}/eigenvecs_QH_cart.npy".format(path),eigenvecs)

    return eigenvals, eigenvecs

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

def get_QH_QM_entropy(omegas):
    """Determine the quantum mechanical, quasi-harmonic estimate for entropy.

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
        A quasi-harmonic estimation of entropy in J / mol K.
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

    return kb * entropy * 1000

def get_QH_CL_entropy(omegas):
    """Determine the quantum mechanical, quasi-harmonic estimate for entropy.

    Uses the angular frequencies calculated from the covariance matrix to
    determine the QH entropy using the classical formulation for a harmonic
    oscillator. Units for constants are given in GROMACS units.

    Parameters
    ----------
    omegas : float list
        The angular frequencies determined by mass-weighted principle
        component analysis.

    Returns
    -------
    entropy : float
        A quasi-harmonic estimation of entropy in J / mol K.
    classical_limit : int
        The (zeroth-indexed) highest mode which can be included in the
        classical limit.
    """
    hbar = 0.0635077993 # kJ / mol ps
    kb = 8.3144621 * 10 ** (-3) # kJ / mol / K
    temp = 298 # K
    entropy = 0 # kJ / mol / K
    classical_range = True
    if len(omegas) == 1:
        classical_limit = None

    for i in range(len(omegas)):
        boltz = hbar * omegas[i] /(kb * temp)
        if boltz >= 1 and classical_range and len(omegas) > 1:
            print("The classical range is valid up to mode {0}".format(i))
            classical_range = False
            classical_limit = i - 1
        entropy += 1 - np.log(boltz)

    return kb * entropy * 1000, classical_limit

def get_convergence(trajs, mass_matrix, times, path, BAT=False):
    """Get quantum mechanical entropy as a function of simulation time.

    Parameters
    ----------
    traj : ndarray
        The full trajectory in cartesian coordinates.
    mass_matrix : ndarray
        A diagonal matrix of size 3N x 3N, where the diagonal elements are
        atomic masses, in multiples of 3.
    times : ndarray
        The times for which the entropy should be calculated.
    path : str
        The full path to the working directory.

    Returns
    -------
    entropies_QM : ndarray
        The entropies evaluated at each time in times using the quantum
        mechanical formulation of entropy of a harmonic oscillator.

    """
    np_qm_cart_file = "{}/entropies_QM_time.npy".format(path)
    np_qm_bat_file = "{}/entropies_QM_BAT_time.npy".format(path)

    if os.path.exists(np_qm_cart_file) and not BAT:

        entropies_QM = np.load(np_qm_cart_file, allow_pickle=True)

    elif os.path.exists(np_qm_bat_file) and BAT:

        entropies_QM = np.load(np_qm_bat_file, allow_pickle=True)

    else:
        entropies_QM = np.zeros(len(times))
        for i, t in enumerate(times):
            eigvals, _ = diag_covar(trajs[:t,:], mass_matrix, path, BAT)
            omegas = [get_angular_freq(eigval) for eigval in eigvals[:-6]]
            entropies_QM[i] = get_QH_QM_entropy(omegas)

        if BAT:
            np.save(np_qm_bat_file,entropies_QM)
        else:
            np.save(np_qm_cart_file,entropies_QM)

    return entropies_QM

def get_entropies_by_mode(omegas):
    """Get quantum and classical entropy estimates by mode.

    Parameters
    ----------
    omegas : float list
        The (ordered) angular frequencies determined by mass-weighted
        principle component analysis.

    Returns
    -------
    entropy_QM_mode : ndarray
        The entropy contribution along each mode by the quantum mechanical
        estimate.
    entropy_CL_mode : ndarray
        The entropy contribution along each mode by the classical estimate.

    """
    entropy_QM_mode, entropy_CL_mode = [], []
    for omega in omegas:
        entropy_QM_mode.append(get_QH_QM_entropy([omega]))
        entropy_CL_mode.append(get_QH_CL_entropy([omega])[0])

    return entropy_QM_mode, entropy_CL_mode

# Functions related to analysis of the projected trajectory
def project_traj(traj, eigenvecs, mass_root, BAT=False):
    """Project the trajectory onto the eigenvectors.

    Parameters
    ----------
    traj : ndarray
        The full trajectory in cartesian coordinates.
    eigenvecs : ndarray
        The eigenvectors of the covariance matrix.
    mass_root : ndarray
        The square root of the diagonalized mass matrix.

    Returns
    -------
    QH_coords : ndarray
        The trajectory projected onto quasi-harmonic coordinates.

    """
    traj_mean = np.mean(traj, axis=0)
    # Put eigenvectors as rows
    if BAT:
        #QH_coords = eigenvecs.T @ (traj - traj_mean).T
        #QH_coords = QH_coords.T
        QH_coords = (traj - traj_mean) @ eigenvecs
    else:
        QH_coords = eigenvecs.T @ mass_root @ (traj - traj_mean).T
        QH_coords = QH_coords.T
        #QH_coords = mass_root @ (traj - traj_mean) @ eigenvecs.T

    print("The QH projected trajectory shape is {}".format(QH_coords.shape))

    return QH_coords

def get_num_PCs(eigenvalues, goal_ratio=0.80):
    """Find the number of eigenvalues needed to capture most of the variance.

    Parameters
    ----------
    eigenvalues : float list
        The eigenvalues determined from PCA represnt the variance of each PC.
    goal_ratio : float
        A number between 0 and 1, representing the ratio of total variance
        captured in the first i principle components.

    Returns
    -------
    num_PCs : float
        The number of PCs needed to encompass 70%(or ratio) of the variance.

    """
    var_ratio = 0
    ind = 0
    total = sum(eigenvalues)
    while var_ratio < goal_ratio:
        ind += 1
        var_ratio = np.sum(eigenvalues[:ind]) / total

    return ind

def correlation_plots(QH_coords, res, n_dim, path, BAT=False):
    """Make 2D correlation plots of the principle components.

    There is a separate plot for each combination of 2 components, capturing 70%
    of the variance in the data. The images are assembled in a .gif to make a
    small movie.

    Parameters
    ----------
    traj : ndarray
        The full trajectory in BAT coordinates.
    eigenvecs : ndarray
        The eigenvectors of the covariance matrix.
    n_dim : float
        The number of PCs needed to encompass 70%(or ratio) of the variance.
    path : str
        The full path to the working directory.

    Returns
    -------
    None.

    """
    print("Using the first {} principle componenents".format(n_dim))

    if BAT:
        png_dir = '{}/PC_analysis_BAT'.format(path)
    else:
        png_dir = '{}/PC_analysis_cart'.format(path)
    for f in os.listdir(png_dir):
        os.remove("{0}/{1}".format(png_dir,f))

    # Make figures of the 2D correlation plots
    for i in range(n_dim):
        for j in range(n_dim - 1):
            if i > j:

                # Scatter plot of 2 principle components
                fig, ax = plt.subplots(figsize=(10,10), facecolor="#F1F0F0",\
                                         constrained_layout=True)
                ax.scatter(QH_coords[:,j], QH_coords[:,i])
                ax.set_xlabel("PC {}".format(j+1), fontdict={"size" : 22})
                ax.set_ylabel("PC {}".format(i+1), fontdict={"size" : 22})
                plt.title("Component #{0} vs component #{1}".format(j+1,i+1),\
                          fontdict={"size" : 24})
                ax.tick_params(axis='y', labelsize=14, direction='in', \
                               width=2, length=5, pad=10)
                ax.tick_params(axis='x', labelsize=14, direction='in', \
                               width=2, length=5, pad=10)
                for k in ["top","bottom","left","right"]:
                    ax.spines[k].set_linewidth(2)
                ax.grid(True)

                # Add text boxes
                # j_DOFs = get_major_DOFs(list(eigenvecs[j]), inds, atoms)
                # j_label = "Major DOFs for PC {0}\n{1}\n{2}\n{3}".format(j+1,
                #             j_DOFs[0], j_DOFs[1], j_DOFs[2])
                # ax.text(0.96, 0.05, j_label, color='black', fontsize=18, \
                #     va="bottom", transform=ax.transAxes, ha="right", \
                #     bbox=dict(facecolor='gainsboro', edgecolor='dimgrey', \
                #     boxstyle='round,pad=0.5', alpha=0.7))
                # i_DOFs = get_major_DOFs(list(eigenvecs[i]), inds, atoms)
                # i_label = "Major DOFs for PC {0}\n{1}\n{2}\n{3}".format(i+1,
                #             i_DOFs[0], i_DOFs[1], i_DOFs[2])
                # ax.text(0.05, 0.95, i_label, color='black', fontsize=18, \
                #     va="top", transform=ax.transAxes, ha="left", \
                #     bbox=dict(facecolor='gainsboro', edgecolor='dimgrey', \
                #     boxstyle='round,pad=0.5', alpha=0.7))

                if BAT:
                    plt.savefig("{0}/PC_analysis_BAT/PC{1}_PC{2}_2Dplot.png"\
                            .format(path, j+1, i+1))
                else:
                    plt.savefig("{0}/PC_analysis_cart/PC{1}_PC{2}_2Dplot.png"\
                            .format(path, j+1, i+1))
                plt.close(fig)

    # Make a gif of the correlations
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('2Dplot.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave("{}/movie.gif".format(png_dir), images, duration=2)

    return None

def analyze_eigenvecs(eigenvecs, atoms, path):

    internal_inds = np.load("{}/internal_inds.npy".format(path), allow_pickle=True)

    eigenvecs = eigenvecs.T
    for eigenvec in eigenvecs[:6,:]:
        #print(eigenvec)
        # Get the top 5 values
        top3 = np.argsort(np.abs(eigenvec))[-3:][::-1]
        print(eigenvec[top3])

        for ind in internal_inds[top3]:
            int_coord = [atoms[i+1].name for i in ind]
            print(int_coord)
        print("\n")

    return None

# Functions related to corrections
def get_anharmonic_corr(QH_coords, eigenvals, entropy_CL_mode, modes, temp, path, BAT=False):
    """Get the anharmonic correction to the (classical) entropy.

    Some of the error in the entropy is due to anharmonicity. Although this
    effect is small (~3% error), it is worth considering. The anharmonic
    correction is actually a correction to the classical entropy, so it is not
    as relevant for the higher modes when correcting the quantum formulation of
    the entropy. In any case, the higher modes are very close to Gaussian
    behavior anyway. This function makes plots to help visualize the impact
    of anharmonicity. See DOI: https://doi.org/10.1021/ct900373z equation (18).

    Parameters
    ----------
    QH_coords : ndarray
        The trajectory projected onto quasi-harmonic coordinates.
    eigenvals : ndarray
        The eigenvalues of the covariance matrix.
    entropy_CL_mode : ndarray
        The classical entropy evaluated by by the eigenmode.
    modes : int
        The number of eigenmodes to apply the anharmonic correction to.
    temp : float
        The temperature of the simulation.
    path : str
        The full path to the working directory.

    Returns
    -------
    anharm_correction : (float)
        The anharmonic correction to the (classical) entropy, up to a certain
        number of eigenmodes in J / mol K.

    """
    # Plots to visualize the extent of anharmonicity
    corr_coeff = plot_anharmonic_regression(QH_coords, eigenvals, path, BAT)
    plot_QH_mode_distribution(QH_coords, eigenvals, corr_coeff, path, BAT)

    #exit(1)

    # bin widths for each mode
    n_frames = np.size(QH_coords[:,0])
    kappa_1D = np.exp(0.5 * (1 + np.log(2 * np.pi / n_frames)))
    bin_widths = get_bin_width(eigenvals, kappa_1D)

    kb = 8.3144621 * 10 ** (-3) # kJ / mol / K
    h = 0.399031271 # kJ / mol * ps
    anharm_mode = []
    anharm_correction = 0

    # Get the anharmonic correction over each mode
    if not BAT:
        QH_coords = QH_coords[:,:-6]
    for mode in range(len(QH_coords[0,:])):
        QH_coord = QH_coords[:, mode]
        bin_width = bin_widths[mode]

        # determine the bin edges for the mode
        bin_edges = [np.min(QH_coord)]
        while bin_edges[-1] < np.max(QH_coord):
            bin_edges.append(bin_edges[-1] + bin_width)

        # Get the probability distribution of the mode
        hist, _ = np.histogram(QH_coord, bins=bin_edges, density=True)

        # Evalute the integral using numeric integration
        integral = 0
        for height in hist:
            if height <= 0:
                continue
            integral += height * bin_width * np.log(height * bin_width)
        integral = integral - np.log(bin_width)

        # Determine the contribution of each mode to the total anharmonic
        # component of the classical entropy
        s_anh = kb * 0.5 * (1 - np.log(h ** 2 / (2 * np.pi * kb * temp))) \
                - kb * integral
        anharm_mode.append(s_anh)

        if mode <= modes:
            anharm_correction += anharm_mode[mode] - \
                                    entropy_CL_mode[mode] / 1000

    # Convert the overall correction to J, and round to nearest decimal point
    anh_corr_J = np.round(anharm_correction * 1000, 1)
    print("The anharm corr: {} J / K mol".format(anh_corr_J))

    return anh_corr_J, anharm_mode

def get_pairwise_corr(QH_coords, eigenvals, anharm_mode, classical_limit, temp, path):
    """Get the pairwise supralinear correction to the QH entropy.

    This correction is based on the supralinear correlations of the linearly
    uncorrelated modes of the projected trajectory.

    Parameters
    ----------
    QH_coords : ndarray
        The trajectory projected onto quasi-harmonic coordinates.
    eigenvals : ndarray
        The eigenvalues of the covariance matrix.
    anharm_mode : ndarray
        The anharmonic contribution to entropy for each eigenmode.
    modes : int
        The number of eigenmodes to apply the anharmonic correction to.
    temp : float
        The temperature of the simulation.
    path : str
        The full path to the working directory.

    Returns
    -------
    pairwise_correction : (float)
        The pairwise correction to the (classical) entropy, up to a certain
        number of eigenmodes in J / mol K.
    """
    # Get bin widths for each mode
    n_frames = len(QH_coords[:,0])
    kappa_2D = np.exp(0.5 * (1 + np.log(2 * np.pi / np.sqrt(n_frames))))
    bin_widths = get_bin_width(eigenvals, kappa_2D)

    kb = 8.3144621 * 10 ** (-3) # kJ / mol / K
    h = 0.399031271 # kJ / mol * ps
    pairwise_mn = np.zeros((classical_limit, classical_limit))
    pairwise_correction = 0
    constant = 1 - np.log(h ** 2 / (2 * np.pi * kb * temp))

    for m in range(classical_limit):
        for n in range(m + 1, classical_limit):

            QH_coord_m = QH_coords[:, m]
            QH_coord_n = QH_coords[:, n]

            # determine the bin edges for the mode
            bin_edges_m = [np.min(QH_coord_m)]
            while bin_edges_m[-1] < np.max(QH_coord_m):
                bin_edges_m.append(bin_edges_m[-1] + bin_widths[m])
            bin_edges_n = [np.min(QH_coord_n)]
            while bin_edges_n[-1] < np.max(QH_coord_n):
                bin_edges_n.append(bin_edges_n[-1] + bin_widths[n])

            # Get the probability distribution of the mode
            hist2d, _, _ = np.histogram2d(QH_coord_m, QH_coord_n, \
                            bins=[bin_edges_m, bin_edges_n], density=True)

            # Evalute the integral using numeric integration
            integral = 0
            for height in hist2d.flatten():
                if height <= 0:
                    continue
                prob = height * bin_widths[m] * bin_widths[n]
                integral += prob * np.log(prob)
            integral_corr = -integral + np.log(bin_widths[m] * bin_widths[n])

            corr = kb * constant - kb * integral_corr - anharm_mode[m] \
                                                - anharm_mode[n]

            #print(kb * constant + kb * integral, - anharm_mode[m] - anharm_mode[n])
            if corr < 0:

                pairwise_mn[m, n] = corr * 1000
                pairwise_correction += corr

    # Convert the overall correction to J, and round to nearest decimal point
    pc_corr_J = np.round(pairwise_correction * 1000, 1)
    print("The supralinear corr: {} J / K mol, with kappa {}\n".format(\
                                    pc_corr_J, np.round(kappa_2D, 3)))

    return pc_corr_J, pairwise_mn

def plot_anharmonic_regression(QH_coords, eigenvals, path, BAT=False):
    """Make a plot of the expected vs. actual probability distribution

    Ideally a plot of the Gaussian distribution against the actual distribution
    should be a straight line which passes through the origin. The correlation
    coefficient provides a measure for how close the low frequency modes are to
    the model distributions.

    Parameters
    ----------
    QH_coords : ndarray
        The trajectory projected onto quasi-harmonic coordinates.
    eigenvals : ndarray
        The eigenvalues of the covariance matrix.
    path : str
        The full path to the working directory.

    Returns
    -------
    None.

    """
    fig, axes = plt.subplots(5, 2, figsize=(12,6), facecolor='gainsboro',\
                             constrained_layout=True)

    # Select several low modes, and one high mode
    select_coords = [0,1,2,3,4,5,6,7,19,49]
    corr_coeff = []

    for ax, i in zip(axes.flat, select_coords):

        # The actual distribution
        actual_dist, bins = np.histogram(QH_coords[:,i], bins=100, \
                                         density=True)

        # The ideal Gaussian distribution
        ave_coord = np.mean(QH_coords[:,i])
        variance = eigenvals[i]
        bins = bins[:-1] + (bins[1] - bins[0]) / 2
        gauss_dist = gauss(bins, ave_coord, variance)

        # The correlation coefficient of a linear regression
        r = np.corrcoef(actual_dist, gauss_dist)[0,1]
        corr_coeff.append(np.round(r,2))

        # Plot
        ax.scatter(gauss_dist, actual_dist, color ="sandybrown", \
                   label="m = {}\nR = {}".format(i+1, np.round(r,2)))
        ax.grid()
        ax.legend()

    if BAT:
        plt.savefig("{0}/anharmonic_linear_reg_BAT.png".format(path))
    else:
        plt.savefig("{0}/anharmonic_linear_reg.png".format(path))
    plt.close(fig)

    return corr_coeff

def plot_QH_mode_distribution(QH_coords, eigenvals, corr_coeff, path, BAT=False):
    """Visualize the probability distribution of the QH coordinates.

    Parameters
    ----------
    QH_coords : ndarray
        The trajectory projected onto quasi-harmonic coordinates.
    eigenvals : ndarray
        The eigenvalues of the covariance matrix.
    corr_coeff : float
        The correlation coefficient from the linear regression of the actual
        distribution against the ideal Gaussian distribution.
    path : str
        The full path to the working directory.

    Returns
    -------
    None

    """
    fig, axes = plt.subplots(5,2,figsize=(16,12), facecolor="#F1F0F0",\
                             constrained_layout=True)
    select_coords = [0,1,2,3,4,5,6,7,19,49]

    for ax, i, j in zip(axes.flat, select_coords, range(len(select_coords))):
        ave_coord = np.mean(QH_coords[:,i])
        variance = eigenvals[i]
        #variance = np.var(QH_coords[:,i])
        spread = np.linspace(ave_coord - 3 * np.sqrt(variance), \
                             ave_coord + 3 * np.sqrt(variance), num=100)
        gauss_func = gauss(spread, ave_coord, variance)
        ax.hist(QH_coords[:,i], bins=1000, density=True, color="#1D8D90", \
                label=r"m = {0}""\n"r"$R^2$ = {1}".format(i+1, np.round(corr_coeff[j]**2,2)))
        ax.plot(spread, gauss_func, linestyle="--", color="darkslategrey")
        ax.legend(fontsize=16)
        ax.grid()

        ax.set_xlabel(r"$a_{{\mathbf{{q}},{{{0}}}}}$".format(i+1), fontdict={"size" : 18})
        ax.set_ylabel(r"$p_{{}}(a_{{\mathbf{{q}},{{{0}}}}})$".format(i+1), fontdict={"size" : 18})

    if BAT:
        plt.savefig("{0}/projected_coords_prob_dist_BAT.png".format(path))
    else:
        plt.savefig("{0}/projected_coords_prob_dist.png".format(path))
    plt.close(fig)

    return None

def gauss(x, mean, variance):
    """Generate an ideal Gaussian distribution.

    Parameters
    ----------
    x : ndarry
        The input values for the Gaussian function.
    mean : float
        The mean for the Gaussian function, which should be near zero.
    variance : float
        The variance of the Gausssian, which is equivalent to the standard
        deviation squared. It is also equal to the eigenvalue of the mode
        which the Gaussian function should model.

    Returns
    -------
    gaussian : ndarray
        The output for the ideal Gaussian distribution with the given mean and
        variance.

    """
    coeff = 1 / np.sqrt(2 * np.pi * variance)
    gaussian = coeff * np.exp(-0.5 * ((x - mean) ** 2 / variance))

    return gaussian

def plot_pairwise_corrs(pairwise_mn, modes, path):
    """Make a 2D color plot of pairwise input to the supralinear correction.
    """
    fig, ax = plt.subplots(figsize=(8,6), facecolor="#F1F0F0",\
                             constrained_layout=True)
    font = {'color': 'black', 'weight': 'semibold', 'size': 22}

    n, m = np.mgrid[0.5:(modes + 1), 0.5:(modes + 1)]
    c = ax.pcolor(m, n, pairwise_mn, cmap=plt.cm.bone)
    cbar = fig.colorbar(c, ax=ax)

    ax.set_xlabel("m", fontdict=font, labelpad=5)
    ax.set_ylabel("n", fontdict=font, labelpad=5)
    plt.xticks(np.arange(2, modes + 1,2))
    plt.yticks(np.arange(2, modes + 1,2))
    cbar.ax.set_ylabel(r"$\Delta s_{mn}^{pc}$ (J / mol K)", fontdict=font, \
                       labelpad=5)
    cbar.ax.tick_params(axis='y', labelsize=18, direction='inout', width=2, \
                        length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    cbar.outline.set_linewidth(2)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)

    plt.savefig("{}/pairwise_correction.png".format(path))
    plt.close(fig)

    return None

def get_bin_width(eigenvals, kappa):
    """Chooses reasonable bin width values for each eigenmode.

    The choice of bin width is based on the individual variances of the
    eigenmodes and a parameter kappa, which is a function of the trajectory
    length and dimension of the bin. The approach is from appendix C of
    source: R Baron, Estimating the configurational entropy from molecular
    dynamics simulations: anharmonicity and correlation corrections to the
    quasi-harmonic approximation. 2006

    Parameters
    ----------
    eigenvals : ndarray
        The eigenvalues of the covariance matrix.
    traj_len : int
        Number of frames in the trajectory.
    dim : int
        The dimension of the bin width can be either 0 or 1.

    Returns
    -------
    bins : ndarray
        The bin widths to use for probability density estimation for each mode.

    """
    bins = kappa * np.sqrt(eigenvals)
    return bins

def opt_bin_width(eigenvals, QH_coords, path, BAT=False):
    """Make a plot to determine an appropiate bin width constant.

    Parameters
    ----------
    eigenvals : ndarray
        The eigenvalues of the covariance matrix.
    QH_coords : ndarray
        The trajectory projected onto quasi-harmonic coordinates.
    path : str
        The full path to the working directory.

    Returns
    -------
    None

    """
    fig, ax = plt.subplots(figsize=(12,6), facecolor="#F1F0F0",\
                             constrained_layout=True)
    font = {'color': 'black', 'weight': 'semibold', 'size': 22}
    n_frames = len(QH_coords[:,0])
    kappa_0 = np.exp(0.5 * (1 + np.log(2 * np.pi / n_frames)))
    kappa = [1/100000, 1/10000, 1/1000, 1/500, 1/200, 1/100, 1/50,\
             1/20, 1/10, 1/5, 1/2, 1, 2, 5, 10, 20, 50, kappa_0]
    markers = ["o", "D", "^"]
    colors = ["dimgray", "darkgray", "lightgray"]
    for i, mode in enumerate([0, 9, 19]):

        QH_coord = QH_coords[:, mode]
        integrals = []

        for k in kappa:
            bin_widths = get_bin_width(eigenvals, k)

            # determine the bin edges for the mode
            bin_edges = [np.min(QH_coord)]
            while bin_edges[-1] < np.max(QH_coord):
                bin_edges.append(bin_edges[-1] + bin_widths[mode])

            # Get the probability distribution of the mode
            hist, _ = np.histogram(QH_coord, bins=bin_edges, density=True)

            # Evalute the integral using numeric integration
            integral = 0
            for height in hist:
                if height <= 0:
                    continue
                integral += (height * np.log(height) * bin_widths[0])
            integrals.append(integral)

        ax.scatter(kappa[:-1], integrals[:-1], marker=markers[i], color=colors[i],\
                   label="m = {}".format(mode+1))
        ax.plot(kappa[-1], integrals[-1], marker=markers[i], color=colors[i],\
                   fillstyle="none", markersize=10)

    ax.set_xscale("log")
    ax.legend(fontsize=22)
    ax.grid()
    ax.set_xlabel(r"$\kappa _1$", fontdict=font, labelpad=5)
    ax.set_ylabel("1D integral", fontdict=font, labelpad=5)
    ax.tick_params(axis='x', labelsize=20, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='y', labelsize=20, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)

    if BAT:
        plt.savefig("{}/optimize_bin_width_BAT.png".format(path))
    else:
        plt.savefig("{}/optimize_bin_width.png".format(path))
    plt.close(fig)

    return None

def opt_bin_width2D(eigenvals, QH_coords, path, BAT=False):
    """Make a plot to determine an appropiate bin width constant.

    Parameters
    ----------
    eigenvals : ndarray
        The eigenvalues of the covariance matrix.
    QH_coords : ndarray
        The trajectory projected onto quasi-harmonic coordinates.
    path : str
        The full path to the working directory.

    Returns
    -------
    None

    """
    fig, ax = plt.subplots(figsize=(12,6), facecolor="#F1F0F0",\
                             constrained_layout=True)
    font = {'color': 'black', 'weight': 'semibold', 'size': 22}
    n_frames = len(QH_coords[:,0])
    kappa_0 = np.exp(0.5 * (1 + np.log(2 * np.pi / np.sqrt(n_frames))))
    kappa = [1/500, 1/200, 1/100, 1/50, 1/20, 1/10, 1/5, 1/2, 1, 2, 5, \
             10, 20, kappa_0]
    markers = ["o", "D", "^"]
    colors = ["dimgray", "darkgray", "lightgray"]
    for i, mode in enumerate([1, 9, 19]):

        QH_coord_m = QH_coords[:, 0]
        QH_coord_n = QH_coords[:, mode]
        integrals = []

        for k in kappa:

            bin_widths = get_bin_width(eigenvals, k)

            # determine the bin edges for the mode
            bin_edges_m = [np.min(QH_coord_m)]
            while bin_edges_m[-1] < np.max(QH_coord_m):
                bin_edges_m.append(bin_edges_m[-1] + bin_widths[0])
            bin_edges_n = [np.min(QH_coord_n)]
            while bin_edges_n[-1] < np.max(QH_coord_n):
                bin_edges_n.append(bin_edges_n[-1] + bin_widths[mode])

            # Get the probability distribution of the mode
            hist2d, _, _ = np.histogram2d(QH_coord_m, QH_coord_n, \
                            bins=[bin_edges_m, bin_edges_n], density=True)

            # Evalute the integral using numeric integration
            integral = 0
            for height in hist2d.flatten():
                if height <= 0:
                    continue
                integral += (height * np.log(height) * bin_widths[0] \
                                                     * bin_widths[mode])
            integrals.append(integral)

        ax.scatter(kappa[:-1], integrals[:-1], marker=markers[i], color=colors[i],\
                   label="m, n = (1, {})".format(mode+1))
        ax.plot(kappa[-1], integrals[-1], marker=markers[i], color=colors[i],\
                   fillstyle="none", markersize=10)

    # Figure settings
    ax.set_xscale("log")
    ax.legend(fontsize=18)
    ax.grid()
    ax.set_xlabel(r"$\kappa _2$", fontdict=font, labelpad=5)
    ax.set_ylabel("2D integral", fontdict=font, labelpad=5)
    ax.tick_params(axis='x', labelsize=20, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='y', labelsize=20, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)

    # Save figure for cart or internal coordinate system
    if BAT:
        plt.savefig("{}/optimize_bin_width_2D_BAT.png".format(path))
    else:
        plt.savefig("{}/optimize_bin_width_2D.png".format(path))
    plt.close(fig)

    return None

# Functions related to grabbing data for analysis
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

def read_topol(path):
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
        with open("{}/topol.top".format(path), "r") as file:
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

def get_combined_traj(path, res):
    """Loads and combines multiple simulation runs into one trajectory.

    Uses PyEmma to load the simulation from each 1us run as an ndarray and then
    concatenates them into one long, 4 us trajectory. Importantly, the
    trajectories have been fitted to the same reference structure and the
    rotational and translational DOFs wrt the first peptide bond have all been
    removed. If the trajectories have already been concatenated, load the
    trajectory from a NumPy file.

    Parameters
    ----------
    path : str
        The full path to the working directory.
    res : str
        The three letter code for the residue.

    Returns
    -------
    trajs : ndarray
        The 4 concatenated trajectories.

    """
    if os.path.exists("{}/combined_traj.npy".format(path)):

        trajs = np.load("{}/combined_traj.npy".format(path), allow_pickle=True)

    else:

        for run in range(0,4):

            # File names for the trajectory and topology
            traj_file = "{0}/run_{2}/{1}_1us_run{2}_fitted.xtc".format(path,res,run)
            top = "{0}/run_0/{1}_1us_run0.gro".format(path,res)

            # Load in trajectory using PyEmma
            traj = pyemma.coordinates.load(traj_file, top=top, features=None)

            if run == 0:
                trajs = traj
            else:
                trajs = np.concatenate((trajs,traj), axis=0)

        np.save("{}/combined_traj.npy".format(path), trajs)

    return trajs

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

def get_mass_mtx(path):
    """Use the topology to get the ordered mass matrix.

    The mass matrix is of size 3N, with each mass repeated 3 times for the x,
    y and z directions, all along the matrix diagonals. All other matrix values
    are zero.

    Parameters
    ----------
    path : str
        The full path to the working directory to load the topology or the
        mass matrix.

    Returns
    -------
    mass_matrix : ndarray
        A 3N x 3N diagonalized array of the masses, repeated in multiples of 3
        for the 3 DOFs.
    """
    mass_mtx_file = "{}/mass_matrix.npy".format(path)

    if False:#os.path.exists(mass_mtx_file):

        mass_matrix = np.load(mass_mtx_file, allow_pickle=True)

    else:

        # read in relevant lines from the topology file, add to dictiory "atoms"
        topol = read_topol("{}/run_0".format(path))
        atoms = dict()
        for atom in topol:
            atoms[int(atom[0])] = Atom(atom)

        # Get the diagonalized mass matrix with dim (3N x 3N)
        mass_matrix = diagonalize_mass(atoms)
        if path != None:
            np.save("{}/mass_matrix.npy".format(path), mass_matrix)

    return mass_matrix, atoms

# Other functions related to the visualization of data
def summarize_results(entropy_QH_QM, anharmonic_corrs, pairwise_corrs, home_dir, BAT=False):
    """Print the key outcomes to file.

    Write to file the main results for the absolute QH entropy, the anharmonic
    correction, entropy differences and the corrected entropy values.

    Parameters
    ----------
    entropy_QH_QM : (float) dict
        A dictionary by residue of the absolute quasi-harmonic estimation of
        entropy in J / mol K.
    anharmonic_corrs : (float) dict
        A dictionary by residue of the anharmonic correction to the QH entropy
        estimation, given in J / K mol.
    pairwise_corrs :
    home_dir : str
        The path to the home working directory.

    Returns
    -------
    None

    """
    spacer = "~" * 25
    if BAT:
        results_file = "{}/entropy_BAT_QH.out".format(home_dir)
    else:
        results_file = "{}/entropy_cart_QH.out".format(home_dir)
    with open(results_file, "w") as f:
        for res in entropy_QH_QM.keys():

            # Determine corrected and relative values
            Uncorrected = np.round(entropy_QH_QM[res], 1)
            ETG_entropy = np.round(entropy_QH_QM["ETG"], 1)
            ETG_entropy_corr = entropy_QH_QM["ETG"] + \
                        anharmonic_corrs["ETG"] + pairwise_corrs["ETG"]
            entropy_corr = Uncorrected + anharmonic_corrs[res] + pairwise_corrs[res]
            entropy_diff = Uncorrected - ETG_entropy
            entropy_diff_corr =  entropy_corr - ETG_entropy_corr

            rel_anh = np.absolute(np.round(anharmonic_corrs[res] / \
                                           entropy_QH_QM[res] * 100, 0))
            rel_pair = np.absolute(np.round(pairwise_corrs[res] / \
                                           entropy_QH_QM[res] * 100, 0))

            f.write(spacer + res + spacer + "\n")
            f.write("Uncorrected absolute entropy: {0} J / mol K\n\t"
                    "Anharmonic correction: {1} ({2}%)\n\tSupralinear pairwise "\
                    "correction: {3} ({4}%)\n\tCorrected entropy: {5}\n"\
                    .format(Uncorrected, anharmonic_corrs[res],\
                    rel_anh, pairwise_corrs[res], rel_pair, entropy_corr))
            if res != "ETG":
                print("The corrected entropy difference for {0} is {1} J / "\
                        "mol K".format(res, entropy_diff_corr))
                f.write("Uncorrected entropy difference: {0} J / mol K\n\t"
                        "Corrected difference : {1} J / mol K\n"\
                            .format(np.round(entropy_diff, 1), \
                                    np.round(entropy_diff_corr, 1)))
            f.write("\n")

    return None

def plot_eigenvals(eigenvalues, path, mw=True, BAT=False):

    fig, ax = plt.subplots(figsize=(16,12), constrained_layout=True)

    # Variables for plot aesthetics
    colors = {"ETG" : "dodgerblue", "E1G" : "darkorange", "E2G" : "deeppink", \
              "E3G" : "blueviolet"}
    font = {'color': 'black', 'weight': 'semibold', 'size': 28}

    i_vals = range(15)

    for res, eigenvals in eigenvalues.items():
        # Make a combined plot
        ax.scatter(i_vals, eigenvals[:15], label=res, linewidth=3, \
                color=colors[res])

    # Plot labels, ticks, borders etc.
    ax.set_ylabel("Eigenvalue", fontdict=font, labelpad=5)
    ax.tick_params(axis='x', labelsize=20, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='y', labelsize=20, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.legend(fontsize=26, loc=4)

    # Save plot to file
    if BAT:
        plt.savefig("{0}/QH_BAT_MW_eigenvalues.png".format(path))
    else:
        plt.savefig("{0}/QH_cart_MW_eigenvalues.png".format(path))
    plt.close(fig)

    return None

def plot_convergence_combined(entropies_QH_QM, times, home_dir, BAT=False):
    """Check the convergence of entropy as a function of time.

    Parameters
    ----------
    entropies_QH_QM : ndarray dict
        The entropies estimated from the quantum mechanical formulation of the
        harmonic oscillator are stored as arrays and accessed using the
        residue (e.g. "ETG") as a key.
    entropies_QH_CL : ndarray dict
        The entropies estimated from the quantum mechanical formulation of the
        harmonic oscillator are stored as arrays and accessed using the
        residue (e.g. "ETG") as a key.
    times : int list
        Times at which the respective entropy estimate was evaluated.
    home_dir : str
        The path to the home working directory.

    Returns
    -------
    None.

    """
    # Make a figure of the convergence
    fig, ax = plt.subplots(figsize=(16,12), constrained_layout=True)

    # Variables for plot aesthetics
    colors = {"ETG" : "dodgerblue", "E1G" : "darkorange", "E2G" : "deeppink", \
              "E3G" : "blueviolet"}
    font = {'color': 'black', 'weight': 'semibold', 'size': 28}

    # Make subplots for each resiude
    for res in entropies_QH_QM.keys():

        # Make a plot with errorbars
        ax.plot(np.divide(times, 1000), entropies_QH_QM[res], linewidth=3, \
                color=colors[res], linestyle="-")
        ax.scatter(np.divide(times, 1000), entropies_QH_QM[res], label=res, \
                linewidth=3, color=colors[res])

    # Plot labels, ticks, borders etc.
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
    if BAT:
        plt.savefig("{}/QH_BAT_convergence.png".format(home_dir))
    else:
        plt.savefig("{}/QH_cart_convergence.png".format(home_dir))
    plt.close(fig)

    return None

def plot_entropies_by_mode_poster(entropy_QM_mode, entropy_CL_mode, home_dir, BAT=False):
    """Plot the entropies as a function of mode.

    Parameters
    ----------
    entropy_QM_mode : ndarray
        The entropy contribution along each mode by the quantum mechanical
        estimate.
    entropy_CL_mode : ndarray
        The entropy contribution along each mode by the classical estimate.
    home_dir : str
        The path to the home working directory.

    Returns
    -------
    None.

    """
    # Make a figure of the convergence
    fig, ax1 = plt.subplots(figsize=(16,12), constrained_layout=True)

    # Make an inset axis, for the entropy by mode
    ax2 = fig.add_axes([0.58,0.14,0.4,0.4])
    ax2.patch.set_alpha(0.75)

    # Variables for plot aesthetics
    colors = {"ETG" : "dodgerblue", "E1G" : "darkorange", "E2G" : "deeppink", \
              "E3G" : "blueviolet"}
    font = {'color': 'black', 'size': 26}

    modes = range(1, len(entropy_QM_mode["ETG"]) + 1)

    # Make subplots for each resiude
    for res in entropy_QM_mode.keys():

        # Evaluate the cummulative entropy by mode
        cummulative_QM, cummulative_CL = [], []
        total_QM, total_CL = 0, 0
        for mode_QM, mode_CL in zip(entropy_QM_mode[res], entropy_CL_mode[res]):
            total_QM += mode_QM
            total_CL += mode_CL
            cummulative_QM.append(total_QM)
            cummulative_CL.append(total_CL)

        # Make a plot of the cummulative entropy
        ax1.plot(modes, cummulative_QM, linewidth=2, color=colors[res], \
                linestyle="-", label=res, alpha=0.6)
        ax1.plot(modes, cummulative_CL, linewidth=2, color=colors[res], \
                linestyle="--", alpha=0.6)

        # Make an inset plot of the entropy contributed by each mode
        ax2.plot(modes, entropy_QM_mode[res], linewidth=2, color=colors[res], \
                linestyle="-", alpha=0.6)
        ax2.plot(modes, entropy_CL_mode[res], linewidth=2, color=colors[res], \
                linestyle="--", alpha=0.6)

    # Plot labels, ticks, borders etc.
    ax1.set_xlabel("Eigenmode", fontdict=font, labelpad=5)
    ax1.set_ylabel(r"Cummulative Entropy (J / mol$\cdot$ K)", fontdict=font, \
                   labelpad=5)
    ax1.tick_params(axis='x', labelsize=18, direction='inout', width=2, \
                    length=5, pad=10)
    ax1.tick_params(axis='y', labelsize=18, direction='inout', width=2, \
                    length=5, pad=10)
    ax2.set_xlabel("Eigenmode", fontdict={"size" : 16}, labelpad=5)
    ax2.set_ylabel(r"Entropy per Eigenmode (J / mol$\cdot$ K)", \
                   fontdict={"size" : 16}, labelpad=5)
    ax2.tick_params(axis='y', labelsize=14, direction='inout', width=2, \
                    length=5, pad=10)
    ax2.tick_params(axis='x', labelsize=14, direction='inout', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax1.spines[i].set_linewidth(2)
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend(fontsize=22, loc=0)

    # Save plot to file
    if BAT:
        plt.savefig("{}/Entropy_by_mode_BAT.png".format(home_dir))
    else:
        plt.savefig("{}/Entropy_by_mode.png".format(home_dir))
    plt.close(fig)

    return None

def plot_entropies_by_mode_poster(entropy_QM_mode, entropy_CL_mode, home_dir, BAT=False):
    """Plot the entropies as a function of mode.

    Parameters
    ----------
    entropy_QM_mode : ndarray
        The entropy contribution along each mode by the quantum mechanical
        estimate.
    entropy_CL_mode : ndarray
        The entropy contribution along each mode by the classical estimate.
    home_dir : str
        The path to the home working directory.

    Returns
    -------
    None.

    """
    # Make a figure of the convergence
    fig, ax1 = plt.subplots(figsize=(16,12), constrained_layout=True)

    # Make an inset axis, for the entropy by mode
    ax2 = fig.add_axes([0.62,0.19,0.35,0.35])
    ax2.patch.set_alpha(0.80)

    # Variables for plot aesthetics
    colors = {"ETG" : "dodgerblue", "E1G" : "darkorange", "E2G" : "deeppink", \
              "E3G" : "blueviolet"}
    font = {'color': 'black', 'size': 34}

    modes = range(1, len(entropy_QM_mode["ETG"]) + 1)

    # Make subplots for each resiude
    for res in entropy_QM_mode.keys():

        # Evaluate the cummulative entropy by mode
        cummulative_QM, cummulative_CL = [], []
        total_QM, total_CL = 0, 0
        for mode_QM, mode_CL in zip(entropy_QM_mode[res], entropy_CL_mode[res]):
            total_QM += mode_QM
            total_CL += mode_CL
            cummulative_QM.append(total_QM)
            cummulative_CL.append(total_CL)

        # Make a plot of the cummulative entropy
        ax1.plot(modes, cummulative_QM, linewidth=3, color=colors[res], \
                linestyle="-", label=res, alpha=0.6)
        ax1.plot(modes, cummulative_CL, linewidth=3, color=colors[res], \
                linestyle="--", alpha=0.6)

        # Make an inset plot of the entropy contributed by each mode
        ax2.plot(modes, entropy_QM_mode[res], linewidth=2, color=colors[res], \
                linestyle="-", alpha=0.6)
        ax2.plot(modes, entropy_CL_mode[res], linewidth=2, color=colors[res], \
                linestyle="--", alpha=0.6)

    # Plot labels, ticks, borders etc.
    ax1.set_xlabel("Eigenmode", fontdict=font, labelpad=5)
    ax1.set_ylabel(r"Cummulative Entropy (J / mol$\cdot$ K)", fontdict=font, \
                   labelpad=5)
    ax1.tick_params(axis='x', labelsize=24, direction='inout', width=2, \
                    length=5, pad=10)
    ax1.tick_params(axis='y', labelsize=24, direction='inout', width=2, \
                    length=5, pad=10)
    ax2.set_xlabel("Eigenmode", fontdict={"size" : 20}, labelpad=5)
    ax2.set_ylabel(r"Entropy per Eigenmode (J / mol$\cdot$ K)", \
                   fontdict={"size" : 20}, labelpad=5)
    ax2.tick_params(axis='y', labelsize=18, direction='inout', width=2, \
                    length=5, pad=10)
    ax2.tick_params(axis='x', labelsize=18, direction='inout', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax1.spines[i].set_linewidth(2)
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend(fontsize=28, loc=0, framealpha=0)

    # Save plot to file
    if BAT:
        plt.savefig("{}/Entropy_by_mode_BAT.png".format(home_dir))
    else:
        plt.savefig("{}/Entropy_by_mode_poster.png".format(home_dir), transparent=True)
    plt.close(fig)

    return None

if __name__ ==  '__main__':
    main(argv)
    exit(0)
