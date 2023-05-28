include("integrators.jl")
include("potentials.jl")
include("diffusionTensors.jl")
include("utils.jl")
include("experiments.jl")
include("dynamics_utils.jl")
import .Integrators: euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D    
import .Potentials: doubleWell1D, LM2013, localWell1D, transformedLocalWell1D, transformedLM2013, transformed2LM2013, softWell1D, transformedSoftWell1D, transformed2SoftWell1D
import .DiffusionTensors: Dconst1D, Dlinear1D, Dquadratic1D
import .Utils: compute_1D_probabilities
import .Experiments: master_1D_experiment, run_1D_experiment_until_given_uncertainty
import .DynamicsUtils: estimate_diffusion_coefficient

# Name
exp_name = "estimating_diffusion_coefficient"

# Integrator Params
T = 100
tau = 1

# Experiment Params
num_repeats = 1
num_step_sizes = 1
integrators = [euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D]
stepsizes = [0.01]

# Histogram parameters
xmin = -5
xmax = 5
n_bins = 30

#Potential and diffusion 
potential = softWell1D
diffusion = Dlinear1D

# Transformations
time_transform = false
space_transform = false
x_of_y = y -> (y/4) * (abs(y) + 4)
checkpoint = true
save_traj = false

# Estimate diffusion coefficient
estimate_diffusion_coefficient = true

# Do not modify below this line ----------------------------------------------
bin_boundaries = range(xmin, xmax, length=n_bins+1)
save_dir = "/home/dominic/JuliaProjects/LangevinIntegrators/outputs/$(exp_name)"

# Run the experiments
@info "Running: $(exp_name)"
for integrator in integrators
master_1D_experiment(integrators, num_repeats, potential, diffusion, T, tau, stepsizes, bin_boundaries, save_dir; chunk_size=10000000, checkpoint=checkpoint, q0=nothing, save_traj=save_traj, time_transform=time_transform, space_transform=space_transform, x_of_y=x_of_y, estimate_diffusion_coefficient=estimate_diffusion_coefficient, segment_length=100)
end
