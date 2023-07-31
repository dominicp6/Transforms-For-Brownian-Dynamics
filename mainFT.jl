include("general_utils/integrators.jl")
include("general_utils/potentials.jl")
include("general_utils/diffusion_tensors.jl")
include("general_utils/probability_utils.jl")
include("experiment_utils/finite_time_experiments.jl")
import .Integrators: euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, leimkuhler_matthews1D, leimkuhler_matthews2D  
import .Potentials: doubleWell1D, LM2013, localWell1D, transformedLocalWell1D, transformedLM2013, transformed2LM2013, softWell1D, transformedSoftWell1D, transformed2SoftWell1D
import .DiffusionTensors: Dconst1D, Dlinear1D, Dquadratic1D
import .ProbabilityUtils: compute_1D_probabilities
import .FiniteTimeExperiments: run_1D_finite_time_convergence_experiment
using WAV

# Name
exp_name = "finite_time_all_untransformed_10M"

# Integrator Params
T =  4         #4.096     # 0.5
ΔT = 0.04    #0.064    # 0.01
tau = 1

# Experiment Params
num_repeats = 10000000
integrators = [euler_maruyama1D, naive_leimkuhler_matthews1D, leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, hummer_leimkuhler_matthews1D]
integrators_transformed = [euler_maruyama1D, naive_leimkuhler_matthews1D, stochastic_heun1D]
reference_integrator = stochastic_heun1D
reference_stepsize = 0.0001
stepsizes = [0.005, 0.010, 0.020, 0.040] #[0.008, 0.016, 0.024, 0.032]
println(stepsizes)

println("Using this many threads: $(Threads.nthreads())")

# Histogram parameters
xmin = -5
xmax = 5
n_bins = 30

#Potential and diffusion 
potential = softWell1D
diffusion = Dlinear1D

# Transformations
untransformed = true
time_transform = false
space_transform = false
x_of_y = y -> (y/4) * (abs(y) + 4)

# Do not modify below this line ----------------------------------------------
bin_boundaries = range(xmin, xmax, length=n_bins+1)
save_dir = "/Users/dominic/JuliaProjects/LangevinIntegrators/new_experiments/$(exp_name)"
#save_dir = "C:/Users/domph.000/JuliaProjects/LangevinIntegrators/outputs/$(exp_name)"
# Run the experiments
@info "Running: $(exp_name)"
run_1D_finite_time_convergence_experiment(integrators, integrators_transformed, reference_integrator, reference_stepsize, num_repeats, potential, diffusion, ΔT, T, tau, stepsizes, bin_boundaries, save_dir; chunk_size=100000, mu0=0, sigma0=1, untransformed=untransformed, time_transform=time_transform, space_transform=space_transform, x_of_y=x_of_y)

# Play notification.wav when done
y, fs = wavread("/Users/dominic/JuliaProjects/LangevinIntegrators/notification.wav")
wavplay(y, fs)