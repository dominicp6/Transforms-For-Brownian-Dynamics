include("general_utils/integrators.jl")
include("general_utils/potentials.jl")
include("general_utils/diffusion_tensors.jl")
include("general_utils/probability_utils.jl")
include("experiment_utils/experiments.jl")
import .Integrators: euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, leimkuhler_matthews1D, leimkuhler_matthews2D  
import .Potentials: doubleWell1D, LM2013, localWell1D, transformedLocalWell1D, transformedLM2013, transformed2LM2013, softWell1D, transformedSoftWell1D, transformed2SoftWell1D
import .DiffusionTensors: Dconst1D, Dabs1D, Dquadratic1D
import .ProbabilityUtils: compute_1D_invariant_distribution
import .Experiments: run_autocorrelation_experiment, run_ess_autocorrelation_experiment

"""
This script computes the autocorrelation function resulting from a numerical approximation of Brownian dynamics
with a certain choice of integrator and timestep. 

Time transformations and Lamperti transforms are supported.
"""

exp_name = "autocorrelation_TT_50K_200r_renorm" # Name
master_dir = "/home/dominic/JuliaProjects/LangevinIntegrators/new_experiments/autocorrelation_exps/" # Directory to save results in


T = 50000               # length of simulation
tau = 1                 # noise coefficient
num_repeats = 200       # number of repeats

# The integrators to use
integrator = stochastic_heun1D
stepsize = 0.01

max_lag = 2000

space_transform=false
time_transform=true

# The potential and diffusion coefficents to use
potential = softWell1D
diffusion = Dabs1D
x_of_y = y -> (y/4) * (abs(y) + 4)  # This spatial transformation is specific to the Dabs1D diffusion coefficient (see paper for details)
                                    # If you want to use a different diffusion coefficient, you will need compute the appropriate mapping from the definition of the Lamperti transform

# Do not modify below this line ----------------------------------------------
save_dir = "$(master_dir)/$(exp_name)"

@info "Running: $(exp_name)"
#run_autocorrelation_experiment(integrator, num_repeats, potential, diffusion, T, tau, stepsize, max_lag, save_dir, space_transform=space_transform, time_transform=time_transform, x_of_y=x_of_y)
run_autocorrelation_experiment(integrator, num_repeats, potential, diffusion, T, tau, stepsize, max_lag, save_dir, space_transform=space_transform, time_transform=time_transform, x_of_y=x_of_y)