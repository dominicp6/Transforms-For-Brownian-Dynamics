include("general_utils/integrators.jl")
include("general_utils/potentials.jl")
include("general_utils/diffusion_tensors.jl")
include("general_utils/probability_utils.jl")
include("experiment_utils/experiments.jl")
import .Integrators: euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, leimkuhler_matthews1D, leimkuhler_matthews2D  
import .Potentials: doubleWell1D, LM2013, localWell1D, transformedLocalWell1D, transformedLM2013, transformed2LM2013, softWell1D, transformedSoftWell1D, transformed2SoftWell1D
import .DiffusionTensors: Dconst1D, Dabs1D, Dquadratic1D
import .ProbabilityUtils: compute_1D_invariant_distribution
import .Experiments: run_autocorrelation_experiment

"""
This script computes the autocorrelation function resulting from a numerical approximation of Brownian dynamics
with a certain choice of integrator and timestep. 

Time transformations and Lamperti transforms are supported.
"""

exp_name = "autocorrelation_test" # Name
master_dir = "/home/dominic/JuliaProjects/LangevinIntegrators/new_experiments" # Directory to save results in


T = 1000             # length of simulation
tau = 1              # noise coefficient
num_repeats = 12

# The integrators to use
integrator = stochastic_heun1D
stepsize = 0.01

# The potential and diffusion coefficents to use
potential = softWell1D
diffusion = Dabs1D

# Do not modify below this line ----------------------------------------------
save_dir = "$(master_dir)/$(exp_name)"

@info "Running: $(exp_name)"
run_autocorrelation_experiment(integrator, num_repeats, potential, diffusion, T, tau, stepsize, save_dir)