include("general_utils/integrators.jl")
include("general_utils/potentials.jl")
include("general_utils/diffusion_tensors.jl")
include("general_utils/probability_utils.jl")
include("experiment_utils/finite_time_experiments.jl")
import .Integrators: euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, leimkuhler_matthews1D, leimkuhler_matthews2D  
import .Potentials: doubleWell1D, LM2013, localWell1D, transformedLocalWell1D, transformedLM2013, transformed2LM2013, softWell1D, transformedSoftWell1D, transformed2SoftWell1D
import .DiffusionTensors: Dconst1D, Dabs1D, Dquadratic1D
import .ProbabilityUtils: compute_1D_invariant_distribution
import .FiniteTimeExperiments: run_1D_finite_time_convergence_experiment

"""
This script performs one-dimensional, variable-diffusion Brownian dynamics experiments and
constructs plots of the error in the finite time distribution, as a function of time, for 
a range of specified step sizes.

Time rescalings and lamperti transforms are supported.
"""

exp_name = "finite_time_test" # Name
master_dir = "path/to/results/directory" # Directory to save results in


T =  4      # length of simulation     
ΔT = 0.04   # interval between snapshots of finite-time distributions
tau = 1     # noise coefficient

num_repeats = 1000

# The integrators to use
integrators = [euler_maruyama1D, naive_leimkuhler_matthews1D, leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, hummer_leimkuhler_matthews1D]
integrators_transformed = [euler_maruyama1D, naive_leimkuhler_matthews1D, stochastic_heun1D]  # integrators to use for transformed potentials, usually a subset
reference_integrator = stochastic_heun1D  # integrator to use for reference solution
reference_stepsize = 0.0001               # stepsize to use for reference solution
stepsizes = [0.005, 0.010, 0.020, 0.040]  # step sizes to use for integrators (>> reference_stepsize)

# The histogram parameters for binning
xmin = -5
xmax = 5
n_bins = 30

# The potential and diffusion coefficents to use
potential = softWell1D
diffusion = Dabs1D

# Information on the transformations
untransformed = true
time_transform = true
space_transform = true
x_of_y = y -> (y/4) * (abs(y) + 4)  # This spatial transformation is specific to the Dabs1D diffusion coefficient (see paper for details)
                                    # If you want to use a different diffusion coefficient, you will need compute the appropriate mapping from the definition of the Lamperti transform
y_of_x = x -> 2 * sign(x) * (sqrt(abs(x) + 1) - 1)  # The inverse of x_of_y

# Do not modify below this line ----------------------------------------------
bin_boundaries = range(xmin, xmax, length=n_bins+1)
save_dir = "$(master_dir)/$(exp_name)"

@info "Running: $(exp_name)"
run_1D_finite_time_convergence_experiment(integrators, integrators_transformed, reference_integrator, reference_stepsize, num_repeats, potential, diffusion, ΔT, T, tau, stepsizes, bin_boundaries, save_dir; chunk_size=100000, mu0=0, sigma0=1, untransformed=untransformed, time_transform=time_transform, space_transform=space_transform, x_of_y=x_of_y, y_of_x=y_of_x)
