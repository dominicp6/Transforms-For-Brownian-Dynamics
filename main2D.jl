include("integrators.jl")
include("potentials.jl")
include("diffusionTensors.jl")
include("utils.jl")
include("experiments.jl")
import .Integrators: euler_maruyamaND, naive_leimkuhler_matthewsND, hummer_leimkuhler_matthewsND, euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D    
import .Potentials: bowl2D, quadrupleWell2D, moroCardin2D, muller_brown
import .DiffusionTensors: Dconst2D, Dlinear2D, Dquadratic2D, DmoroCardin, Doseen, DRinvertible
import .Utils:  compute_2D_probabilities
import .Experiments: master_2D_experiment

# Name
exp_name = "MV_R_LeimkuhlerMatthews"

# Integrator Params
T = 100
tau = 1

# Experiment Params
num_repeats = 11
num_step_sizes = 8
integrators = [naive_leimkuhler_matthewsND]
stepsizes = 10 .^ range(-4.0,stop=-3.2,length=num_step_sizes)
println(stepsizes)
# Histogram parameters
xmin = -5
xmax = 5
ymin = -5
ymax = 5
n_bins = 30

#Potential and diffusion 
potential = quadrupleWell2D
diffusion = DRinvertible

# Transformations
time_transform = false
space_transform = false
x_of_y = y -> (y/4) * (abs(y) + 4)
checkpoint = true
save_traj = false

# Do not modify below this line ----------------------------------------------
save_dir = "/home/dominic/JuliaProjects/LangevinIntegrators/outputs/$(exp_name)"

# Run the experiments
@info "Running: $(exp_name)"
master_2D_experiment(integrators, num_repeats, potential, diffusion, T, tau, stepsizes, xmin, ymin, xmax, ymax, n_bins, save_dir, chunk_size=1000000, checkpoint=false, q0=nothing, save_traj=false)
