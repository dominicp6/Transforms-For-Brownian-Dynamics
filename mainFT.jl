include("integrators.jl")
include("potentials.jl")
include("diffusionTensors.jl")
include("utils.jl")
include("finite_time_experiments.jl")
import .Integrators: euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, leimkuhler_matthews1D, leimkuhler_matthews2D  
import .Potentials: doubleWell1D, LM2013, localWell1D, transformedLocalWell1D, transformedLM2013, transformed2LM2013, softWell1D, transformedSoftWell1D, transformed2SoftWell1D
import .DiffusionTensors: Dconst1D, Dlinear1D, Dquadratic1D
import .Utils: compute_1D_probabilities
import .FiniteTimeExperiments: run_1D_finite_time_convergence_experiment

# Name
exp_name = "finite_time_test_10M"

# Integrator Params
T = 0.5
ΔT = 0.01
tau = 1

# Experiment Params
num_repeats = 10000000
# num_step_sizes = 4
integrators = [euler_maruyama1D, naive_leimkuhler_matthews1D, leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, hummer_leimkuhler_matthews1D]
reference_integrator = stochastic_heun1D
reference_stepsize = 0.0002
stepsizes = [0.0016]#, 0.024, 0.032, 0.048]
println(stepsizes)
# Histogram parameters
xmin = -5
xmax = 5
n_bins = 30

#Potential and diffusion 
potential = softWell1D
diffusion = Dlinear1D


# Transformations
time_transform = true
space_transform = true
x_of_y = y -> (y/4) * (abs(y) + 4)

# Do not modify below this line ----------------------------------------------
bin_boundaries = range(xmin, xmax, length=n_bins+1)
save_dir = "/home/dominic/JuliaProjects/LangevinIntegrators/outputs/$(exp_name)"
#save_dir = "C:/Users/domph.000/JuliaProjects/LangevinIntegrators/outputs/$(exp_name)"
# Run the experiments
@info "Running: $(exp_name)"
run_1D_finite_time_convergence_experiment(integrators, reference_integrator, reference_stepsize, num_repeats, potential, diffusion, ΔT, T, tau, stepsizes, bin_boundaries, save_dir; chunk_size=10, mu0=0, sigma0=1, save_traj=false, time_transform=time_transform, space_transform=space_transform, x_of_y=x_of_y)