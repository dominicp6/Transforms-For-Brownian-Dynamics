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


# Integrator Params
tau = 1

# Experiment Params
num_repeats = 6000

# Histogram parameters
xmin = -5
xmax = 5
n_bins = 30

#Potential and diffusion 
potential = softWell1D
diffusion = Dlinear1D

# Transformations
time_transform = true
space_transform = false
x_of_y = y -> (y/4) * (abs(y) + 4)
checkpoint = false
save_traj = false

# Exp info
bin_boundaries = range(xmin, xmax, length=n_bins+1)

# Untransformed Experiments
integrators = [euler_maruyama1D]#, naive_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D]
target_uncertainties = 10 .^ range(-3.4,stop=-3.4,length=1)

num_step_sizes = 6

LM = [10 .^ range(-2.0, stop=-1.5, length = 5)]
EM = [10 .^ range(-2.95, stop=-2.70, length = 7)]#[10 .^ range(-3.1,stop=-2.45,length=num_step_sizes), 10 .^ range(-3.0,stop=-2.65,length=num_step_sizes), 10 .^ range(-3.1,stop=-2.85,length=num_step_sizes), 10 .^ range(-3.2,stop=-2.95,length=num_step_sizes)]
NLM = [10 .^ range(-1.7,stop=-1.35,length=num_step_sizes), 10 .^ range(-1.75,stop=-1.4,length=num_step_sizes), 10 .^ range(-1.7,stop=-1.5,length=num_step_sizes), 10 .^ range(-1.8,stop=-1.50,length=num_step_sizes), 10 .^ range(-1.9,stop=-1.55,length=num_step_sizes)]
MM = [10 .^ range(-2.85,stop=-2.20,length=num_step_sizes)]#, 10 .^ range(-2.85,stop=-2.5,length=num_step_sizes), 10 .^ range(-3.0,stop=-2.6,length=num_step_sizes), 10 .^ range(-3.0,stop=-2.75,length=num_step_sizes), 10 .^ range(-3.2,stop=-2.825,length=num_step_sizes)]
SH = [10 .^ range(-1.9,stop=-1.65,length=num_step_sizes)]#, 10 .^ range(-1.95,stop=-1.7,length=num_step_sizes), 10 .^ range(-1.95,stop=-1.75,length=num_step_sizes), 10 .^ range(-2.0,stop=-1.8,length=num_step_sizes), 10 .^ range(-2.05,stop=-1.85,length=num_step_sizes)]

# EM = [10 .^ range(-3.1,stop=-2.85,length=num_step_sizes)]

stepsizes_list = [EM]

for (int_idx, integrator) in enumerate(integrators)
    for (unt_idx, target_uncertainty) in enumerate(target_uncertainties)
        exp_name = "$(string(nameof(integrator)))_TT2_$(target_uncertainty)"
        @info "Running: $(exp_name)"
        #save_dir = "C:/Users/domph.000/JuliaProjects/LangevinIntegrators/outputs/$(exp_name)"
        save_dir = "/home/dominic/JuliaProjects/LangevinIntegrators/outputs/$(exp_name)"
        stepsizes = stepsizes_list[int_idx][unt_idx]
        probabilities = compute_1D_probabilities(potential, tau, bin_boundaries)
        run_1D_experiment_until_given_uncertainty(integrator, num_repeats, potential, diffusion, tau, stepsizes, probabilities, bin_boundaries, save_dir, target_uncertainty; chunk_size=500, checkpoint=checkpoint, q0=nothing, save_traj=save_traj, time_transform=time_transform, space_transform=space_transform, x_of_y=x_of_y)
    end
end

