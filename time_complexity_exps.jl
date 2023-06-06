include("integrators.jl")
include("potentials.jl")
include("diffusionTensors.jl")
include("utils.jl")
include("experiments.jl")
import .Integrators: euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, leimkuhler_matthews1D, leimkuhler_matthews2D  
import .Potentials: doubleWell1D, LM2013, localWell1D, transformedLocalWell1D, transformedLM2013, transformed2LM2013, softWell1D, transformedSoftWell1D, transformed2SoftWell1D
import .DiffusionTensors: Dconst1D, Dlinear1D, Dquadratic1D
import .Utils: compute_1D_probabilities
import .Experiments: master_1D_experiment, run_1D_experiment_until_given_uncertainty


# Integrator Params
tau = 1

# Experiment Params
num_repeats = 12

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

# Exp info
bin_boundaries = range(xmin, xmax, length=n_bins+1)

# Untransformed Experiments
integrators = [euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, leimkuhler_matthews1D]
target_uncertainties = 10 .^ range(-3.2,stop=-4,length=5)

num_step_sizes = 2

EM = [10 .^ range(-3.0,stop=-2.5,length=num_step_sizes), 10 .^ range(-3.2,stop=-2.7,length=num_step_sizes), 10 .^ range(-3.4,stop=-2.9,length=num_step_sizes), 10 .^ range(-3.5,stop=-3.0,length=num_step_sizes), 10 .^ range(-3.7,stop=-3.2,length=num_step_sizes)]
# EM = [10 .^ range(-3.7,stop=-3.5,length=num_step_sizes), 10 .^ range(-3.5,stop=-3.0,length=num_step_sizes), 10 .^ range(-3.4,stop=-2.9,length=num_step_sizes), 10 .^ range(-3.2,stop=-2.7,length=num_step_sizes), 10 .^ range(-3.0,stop=-2.5,length=num_step_sizes)]
# NLM = [10 .^ range(-3,stop=-1.5,length=num_step_sizes), 10 .^ range(-3,stop=-1.5,length=num_step_sizes), 10 .^ range(-3,stop=-1.5,length=num_step_sizes), 10 .^ range(-3,stop=-1.5,length=num_step_sizes), 10 .^ range(-3,stop=-1.5,length=num_step_sizes)]
HLM = [10 .^ range(-2.8,stop=-2.3,length=num_step_sizes), 10 .^ range(-2.6,stop=-2.1,length=num_step_sizes), 10 .^ range(-2.4,stop=-1.9,length=num_step_sizes), 10 .^ range(-2.2,stop=-1.7,length=num_step_sizes), 10 .^ range(-2.0,stop=-1.5,length=num_step_sizes)]
MM = [10 .^ range(-3.7,stop=-3.2,length=num_step_sizes), 10 .^ range(-3.5,stop=-3.0,length=num_step_sizes), 10 .^ range(-3.3,stop=-2.8,length=num_step_sizes), 10 .^ range(-3.1,stop=-2.6,length=num_step_sizes), 10 .^ range(-2.9,stop=-2.4,length=num_step_sizes)]
SH = [10 .^ range(-2.8,stop=-2.3,length=num_step_sizes), 10 .^ range(-2.6,stop=-2.1,length=num_step_sizes), 10 .^ range(-2.4,stop=-1.9,length=num_step_sizes), 10 .^ range(-2.2,stop=-1.7,length=num_step_sizes), 10 .^ range(-2.0,stop=-1.5,length=num_step_sizes)]
LM = [10 .^ range(-2.4,stop=-2.2,length=num_step_sizes), 10 .^ range(-2.3,stop=-2.1,length=num_step_sizes), 10 .^ range(-2.2,stop=-2.0,length=num_step_sizes), 10 .^ range(-2.1,stop=-1.9,length=num_step_sizes), 10 .^ range(-2.0,stop=-1.8,length=num_step_sizes)]

stepsizes_list = [EM, HLM, MM, SH, LM]

for (int_idx, integrator) in enumerate(integrators)
    for (unt_idx, target_uncertainty) in enumerate(target_uncertainties)
        exp_name = "$(string(nameof(integrator)))_$(target_uncertainty)"
        @info "Running: $(exp_name)"
        save_dir = "/home/dominic/JuliaProjects/LangevinIntegrators/outputs/$(exp_name)"
        stepsizes = stepsizes_list[int_idx][unt_idx]
        probabilities = compute_1D_probabilities(potential, tau, bin_boundaries)
        run_1D_experiment_until_given_uncertainty(integrator, num_repeats, potential, diffusion, tau, stepsizes, probabilities, bin_boundaries, save_dir, target_uncertainty; chunk_size=500, checkpoint=checkpoint, q0=nothing, save_traj=save_traj, time_transform=time_transform, space_transform=space_transform, x_of_y=x_of_y)
    end
end

