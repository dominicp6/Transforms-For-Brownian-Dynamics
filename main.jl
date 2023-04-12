include("integrators.jl")
include("potentials.jl")
include("diffusionTensors.jl")
include("utils.jl")
include("experiments.jl")
import .Integrators: euler_maruyamaND, naive_leimkuhler_matthewsND, hummer_leimkuhler_matthewsND, euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D
import .Potentials: bowl2D, doubleWell1D, quadrupleWell2D, moroCardin2D, muller_brown, LM2013, localWell1D, transformedLocalWell1D, transformedLM2013, transformed2LM2013, softWell1D, transformedSoftWell1D, transformed2SoftWell1D
import .DiffusionTensors: Dconst1D, Dlinear1D, Dquadratic1D, Dconst2D, Dlinear2D, Dquadratic2D, DmoroCardin, Doseen
import .Utils: compute_1D_probabilities
import .Experiments: master_1D_experiment

# Name
exp_name = "time_transformed_soft_well_1D_10K"

# Integrator Params
T = 10000
tau = 1

# Experiment Params
num_repeats = 12
num_step_sizes = 10
integrators = [euler_maruyama1D, naive_leimkuhler_matthews1D]
stepsizes = 10 .^ range(-3,stop=-1,length=num_step_sizes)

# Histogram parameters
xmin = -5
xmax = 5
n_bins = 30

#Potential and diffusion 
potential = softWell1D
diffusion = Dlinear1D

# Time Transformation
time_transform = true

# Do not modify below this line ----------------------------------------------
bin_boundaries = range(xmin, xmax, length=n_bins+1)
save_dir = "/home/dominic/JuliaProjects/LangevinIntegrators/outputs/$(exp_name)"

# Run the experiments
master_1D_experiment(integrators, num_repeats, potential, diffusion, T, tau, stepsizes, bin_boundaries, save_dir; chunk_size=10000000, checkpoint=false, q0=nothing, save_traj=false, time_transform=time_transform)





# =====================================================
# Define the forward transformation
# y(x) = 2 * sign(x) * (sqrt(1+abs(x))-1)

# x_bins = scale_range(range(xmin,xmax,n_bins), y)

# rb_hist = debias_hist(db_hist, x)