include("../general_utils/integrators.jl")
include("../general_utils/potentials.jl")
include("../general_utils/diffusion_tensors.jl")
include("../general_utils/probability_utils.jl")
include("../experiment_utils/experiments.jl")
import .Integrators: euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, limit_method_with_variable_diffusion1D  
import .Potentials: doubleWell1D, LM2013, localWell1D, transformedLocalWell1D, transformedLM2013, transformed2LM2013, softWell1D, transformedSoftWell1D, transformed2SoftWell1D
import .DiffusionTensors: Dconst1D, Dabs1D, Dquadratic1D
import .ProbabilityUtils: compute_1D_invariant_distribution
import .Experiments: master_1D_experiment, run_1D_experiment_until_given_error


# Integrator Params
tau = 1                 # value of kT (noise amplitude scaling)

# Experiment Params
num_repeats = 500

# Histogram parameters
xmin = -5
xmax = 5
n_bins = 30

#Potential and diffusion 
potential = softWell1D
diffusion = Dabs1D

# Transformations
time_transform = true
space_transform = false
x_of_y = y -> (y/4) * (abs(y) + 4)
checkpoint = false
save_traj = false

# Exp info
bin_boundaries = range(xmin, xmax, length=n_bins+1)

# Untransformed Experiments
integrators = [naive_leimkuhler_matthews1D]

# Set the target uncertanties to investigate 
target_uncertainties = 10 .^ range(-3.0,stop=-3.0,length=1)

num_step_sizes = 6

# Identify a narrow range of step sizes which includes the optimal step size
LM = [10 .^ range(-1.7,stop=-1.0,length=num_step_sizes)]

# In this case, just run for the parameters of the leimkuhler-matthews integrator (extend/modify this list as require)
stepsizes_list = [LM]

for (int_idx, integrator) in enumerate(integrators)
    for (unt_idx, target_uncertainty) in enumerate(target_uncertainties)
        exp_name = "$(string(nameof(integrator)))_$(target_uncertainty)"
        @info "Running: $(exp_name)"
        save_dir = "outputs/$(exp_name)"
        stepsizes = stepsizes_list[int_idx][unt_idx]
        probabilities = compute_1D_invariant_distribution(potential, tau, bin_boundaries)
        run_1D_experiment_until_given_error(integrator, num_repeats, potential, diffusion, tau, stepsizes, probabilities, bin_boundaries, save_dir, target_uncertainty; chunk_size=500, checkpoint=checkpoint, q0=nothing, time_transform=time_transform, space_transform=space_transform, x_of_y=x_of_y)
    end
end

