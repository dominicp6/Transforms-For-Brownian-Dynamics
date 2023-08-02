include("general_utils/integrators.jl")
include("general_utils/potentials.jl")
include("general_utils/diffusion_tensors.jl")
include("general_utils/probability_utils.jl")
include("experiment_utils/finite_time_experiments.jl")
import .Integrators: euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, leimkuhler_matthews1D, leimkuhler_matthews2D  
import .Potentials: doubleWell1D, LM2013, localWell1D, transformedLocalWell1D, transformedLM2013, transformed2LM2013, softWell1D, transformedSoftWell1D, transformed2SoftWell1D
import .DiffusionTensors: Dconst1D, Dabs1D, Dquadratic1D
import .ProbabilityUtils: compute_1D_invariant_distribution
import .FiniteTimeExperiments: run_1D_finite_time_convergence_experiment, finite_time_convergence_to_invariant_measure

"""
This script performs one-dimensional (untransformed), variable-diffusion Brownian dynamics experiments and
constructs plots of the L1 difference between the finite-time distributions and the invariant 
distribution, as a function of time.
"""

exp_name = "convergence_to_invariant_measure_10M" # Name
master_dir = "/home/dominic/JuliaProjects/LangevinIntegrators/new_experiments" # Directory to save results in


T =  4      # length of simulation     
ΔT = 0.04   # interval between snapshots of finite-time distributions
tau = 1     # noise coefficient

num_repeats = 10000000

# The integrators to use
integrator = stochastic_heun1D
stepsize = 0.00001

# The histogram parameters for binning
xmin = -5
xmax = 5
n_bins = 30

# The potential and diffusion coefficents to use
potential = softWell1D
diffusion = Dabs1D

# Do not modify below this line ----------------------------------------------
bin_boundaries = range(xmin, xmax, length=n_bins+1)
save_dir = "$(master_dir)/$(exp_name)"

@info "Running: $(exp_name)"
finite_time_convergence_to_invariant_measure(integrator, stepsize, num_repeats, potential, diffusion, ΔT, T, tau, bin_boundaries, save_dir; chunk_size=2500, mu0=0, sigma0=1)