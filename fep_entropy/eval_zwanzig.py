import sys
import numpy as np
import matplotlib.pyplot as plt
import os

def main(argv):

    try:
        temperature = int(argv[1])
        system = argv[2]
    except IndexError:
        temperature = 298
        system = "ETG-E1G"

    # Variables for describing systems and file names
    coupling_parameters = ["000", "010", "020", "030", "040", "050", "060", \
                            "070", "080", "090", "100"]
    directions = {"forward" : coupling_parameters, \
                  "reverse" : coupling_parameters[::-1]}

    forward_del_F = dict()
    reverse_del_F = dict()
    delta_F = dict()

    if os.path.exists("delta_Us.npy") and os.path.exists("delta_F.npy"):

        delta_Us = np.load("delta_Us.npy", allow_pickle=True).item()
        delta_F = np.load("delta_F.npy", allow_pickle=True).item()

    else:

        delta_Us = dict()
        for d in directions.keys():
            for alch_change in range(len(coupling_parameters)-1):
                ref_state = directions[d][alch_change]
                target_state = directions[d][alch_change + 1]
                ref_file = "{0}/{1}_in_{1}.xvg".format(d, ref_state)
                target_file = "{0}/{1}_in_{2}.xvg".format(d, target_state, \
                                                                    ref_state)
                ref_traj = np.loadtxt(ref_file, comments=["#", "@"])
                target_traj = np.loadtxt(target_file, comments=["#", "@"])

                del_F = free_energy_diff(ref_traj[:,1], target_traj[:,1], d, \
                                                                temperature)
                if d == "forward":
                    key = "{0}->{1}".format(ref_state, target_state)
                    delta_Us[key] = target_traj[:,1] - ref_traj[:,1]
                else:
                    key = "{0}<-{1}".format(target_state, ref_state)
                    delta_Us[key] = ref_traj[:,1] - target_traj[:,1]
                print("State A: {}, State B: {}, deltaF : {}".format(ref_state,\
                            target_state, del_F))
                delta_F[key] = del_F

        np.save("delta_Us.npy", delta_Us)
        np.save("delta_F.npy", delta_F)

    #plot_energy_dist(delta_Us, temperature)

    out_file = "{}_{}K.out".format(system, temperature)
    if os.path.exists(out_file):
        os.remove(out_file)

    total_f, total_r = 0, 0
    with open(out_file, "a") as f:
        f.write("\n~~~~~ Results in the forward direction ~~~~~\n")
        for k, i in delta_F.items():
            if "->" in k:
                total_f += i
                f.write("{} : {} kJ / mol\n".format(k, np.round(i,1)))
        f.write("Total forward: {} kJ / mol\n".format(np.round(total_f,1)))
        f.write("~~~~~ Results in the reverse direction ~~~~~\n")
        for k, i in sorted(delta_F.items()):
            if "<-" in k:
                total_r += i
                f.write("{} : {} kJ / mol\n".format(k, np.round(i,1)))
        f.write("Total reverse: {} kJ / mol\n".format(np.round(total_r,1)))

        ineq_lines = check_GB_ineq(delta_Us)
        for line in ineq_lines:
            f.write(line)

    for k,i in delta_F.items():
        print(k, ":", i)

    plot_dF_lambda(delta_F, system)
    #plot_prob_boltz()

def free_energy_diff(ref_energy, target_energy, direction, temperature):
    """Evaluate the free energy difference between two systems.

    The free energy difference is calculated with respect to a trajectory in
    state A using the Zwanzig expression. To maintian direct comparability of
    the free energy, the direction is accounted for in the calculation of the
    free energy.

    Parameters
    ----------
    ref_energy : np.ndarray
        An array of the total energy calculated for the trajectory with respect
        to system A in kJ / mol.
    target_energy : np.ndarray
        An array of the total energy calculated for the same trajectory, but
        with respect to system B (also in kJ / mol). This is in GROMACS done
        with "gmx mdrun -s target_state.tpr -rerun ref_state.xtc"

    Returns
    -------
    del_F : float
        The free energy difference for the two states in kJ / mol.

    """
    kB_T = 0.0083144621 * temperature
    if direction == "reverse":
        exp_term = np.mean(np.exp((ref_energy - target_energy) / kB_T))
        del_F = kB_T * np.log(exp_term)
    else:
        exp_term = np.mean(np.exp(-(target_energy - ref_energy) / kB_T))
        del_F = - kB_T * np.log(exp_term)

    return del_F

def check_GB_ineq(delta_Us):
    """Check whether the Gibbs-Bogoliubov inequality is held.

    The inequality is based off comparisons of the free energy to the average
    energy difference in each frame of reference. The lower and upper bounds of
    each step are printed as bound on the free energy difference.

    Parameters
    ----------
    delta_Us : (ndarray) dictionary
        A dictionary containing the trajectory of energy differences between
        adjacent steps, for forward and reverse directions.

    Returns
    -------
    None

    """
    ineq_lines = []
    forward_keys = []
    reverse_keys = []
    for key in delta_Us.keys():
        if "->" in key:
            forward_keys.append(key)
        else:
            reverse_keys.append(key)
    forward_keys.sort()
    reverse_keys.sort()

    ineq_lines.append("\nThe Gibbs-Bogoliubov inequalities:\n")
    for f, r in zip(forward_keys, reverse_keys):
        ave_f = np.round(np.mean(delta_Us[f]), 1)
        ave_r = np.round(np.mean(delta_Us[r]), 1)
        ineq_lines.append("For {}, free enegy is bounded as,\n\t{} < Delta A < {}\n"\
            .format(f, ave_r, ave_f))

    return ineq_lines

def plot_energy_dist(delta_Us, temperature):
    """Make a plot of the energy distributions.

    Parameters
    ----------
    ref_energy : np.ndarray
        An array of the total energy calculated for the trajectory with respect
        to system A in kJ / mol.
    target_energy : np.ndarray
        An array of the total energy calculated for the same trajectory, but
        with respect to system B (also in kJ / mol). This is in GROMACS done
        with "gmx mdrun -s target_state.tpr -rerun ref_state.edr"
    key : str
        A label of the systems involved, used for naming the file.

    """
    fig, ax = plt.subplots()
    for key, delta_U in delta_Us.items():
        if "->" in key:
            ax.hist(delta_U, bins=100, density=True, label=key, alpha=0.6)
            ave = np.round(np.mean(delta_U), 1)
            var = np.round(np.var(delta_U), 1)
            print(key, ": ave =", ave, "var =", var)

            spread = np.linspace(ave - 3 * np.sqrt(var), ave + 3 * np.sqrt(var), \
                                    num=100)
            gauss_func = gauss(spread, ave, var)
            ax.plot(spread, gauss_func, linestyle="--", color="red")

    ax.tick_params(axis='y', labelsize=14, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=14, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"$\Delta U$", labelpad=5)
    plt.xlim((-10,10))
    ax.legend()
    plt.show()
    plt.savefig("{}_{}_distributions.png".format("ETG-E1G", temperature))

    return None

def plot_prob_boltz():
    """Make a plot of the probability distributios.
    """
    fig, ax = plt.subplots()

    for i in ["forward", "reverse"]:
        ref_file = "direct/{0}_ref.xvg".format(i)
        target_file = "direct/{0}_target.xvg".format(i)
        ref_traj = np.loadtxt(ref_file, comments=["#", "@"])
        target_traj = np.loadtxt(target_file, comments=["#", "@"])

        del_U = target_traj[:,1] - ref_traj[:,1]
        ax.hist(del_U, bins=100, density=True, label=i, alpha=0.6)

    # Plot settings
    ax.tick_params(axis='y', labelsize=14, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=14, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"$\Delta U$", labelpad=5)
    ax.set_ylabel("Probability", labelpad=5)
    ax.legend()
    plt.show()
    plt.savefig("probability_densities.png")

    return None

def plot_dF_lambda(delta_F, system):
    """Plot the free energy as a function of lambda.

    Parameters
    ----------
    delta_F : (float) dictionary
        The free energy difference between adjacent lambda steps for forward and
        reverse directions.

    Returns
    -------
    None

    """
    forward_dF = []
    reverse_dF = []
    forward = []
    revs = []
    for key in delta_F.keys():
        if "->" in key:
            n = float(key[:3]) / 100
            forward_dF.append(delta_F[key])
            forward.append(n)
        else:
            n = float(key[:3]) / 100
            reverse_dF.append(delta_F[key])
            revs.append(n)
    #forward_dF.sort()
    #reverse_dF.sort()

    fig, ax = plt.subplots()
    ax.scatter(forward, forward_dF, label="forward")
    ax.scatter(revs, reverse_dF, label="reverse")

    # Plot settings
    ax.tick_params(axis='y', labelsize=14, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=14, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"$\lambda$", labelpad=5, fontsize=28)
    ax.set_ylabel(r"$\Delta A$", labelpad=5, fontsize=28)
    ax.legend()
    plt.show()
    plt.savefig("{}_lambda_plot.png".format(system))

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

if __name__ == '__main__':
    main(sys.argv)
