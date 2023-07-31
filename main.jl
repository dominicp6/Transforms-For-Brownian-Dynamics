include("general_utils/integrators.jl")
include("general_utils/potentials.jl")
include("general_utils/diffusion_tensors.jl")
include("general_utils/probability_utils.jl")
include("experiment_utils/experiments.jl")
import .Integrators: euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, leimkuhler_matthews1D, leimkuhler_matthews2D  
import .Potentials: doubleWell1D, LM2013, localWell1D, transformedLocalWell1D, transformedLM2013, transformed2LM2013, softWell1D, transformedSoftWell1D, transformed2SoftWell1D
import .DiffusionTensors: Dconst1D, Dlinear1D, Dquadratic1D
import .ProbabilityUtils: compute_1D_probabilities
import .Experiments: master_1D_experiment, run_1D_experiment_until_given_uncertainty
using WAV

# Name
exp_name = "baoab_limit_method_75M_new"

# Integrator Params
T = 75000000
tau = 1

# Experiment Params
num_repeats = 12
num_step_sizes = 10
integrators = [leimkuhler_matthews1D]
stepsizes = 10 .^ range(-3,stop=-1,length=num_step_sizes)
println(stepsizes)

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

# Do not modify below this line ----------------------------------------------
bin_boundaries = range(xmin, xmax, length=n_bins+1)
save_dir = "/home/dominic/JuliaProjects/LangevinIntegrators/outputs/$(exp_name)"

# Run the experiments
@info "Running: $(exp_name)"
master_1D_experiment(integrators, num_repeats, potential, diffusion, T, tau, stepsizes, bin_boundaries, save_dir; chunk_size=10000000, checkpoint=checkpoint, q0=nothing, save_traj=save_traj, time_transform=time_transform, space_transform=space_transform, x_of_y=x_of_y)

# Play notification.wav when done
y, fs = wavread("/Users/dominic/JuliaProjects/LangevinIntegrators/notification.wav")
wavplay(y, fs)