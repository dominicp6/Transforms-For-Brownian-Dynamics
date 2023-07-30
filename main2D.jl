include("general_utils/integrators.jl")
include("general_utils/potentials.jl")
include("general_utils/diffusion_tensors.jl")
include("general_utils/probability_utils.jl")
include("experiment_utils/experiments2D.jl")
using LinearAlgebra
import .Integrators: euler_maruyama2D, naive_leimkuhler_matthews2D, hummer_leimkuhler_matthews2D, stochastic_heun2D, leimkuhler_matthews2D    
import .Potentials: bowl2D, quadrupleWell2D, moroCardin2D, muller_brown, softQuadrupleWell2D
import .DiffusionTensors: Dconst2D, Dlinear2D, Dquadratic2D, DmoroCardin, Doseen, DRinvertible
import .ProbabilityUtils:  compute_2D_probabilities
import .Experiments2D: master_2D_experiment

# Name
exp_name = "MV_1M_baoab_limit_method"

# Integrator Params
T = 1000000
tau = 1

# Experiment Params
num_repeats = 11
num_step_sizes = 10
integrators = [leimkuhler_matthews2D]
stepsizes = 10 .^ range(-2.5,stop=-0.5,length=num_step_sizes)
println(stepsizes)
# Histogram parameters
xmin = -3
xmax = 3
ymin = -3
ymax = 3
n_bins = 30

#Potential and diffusion 
potential = softQuadrupleWell2D
diffusion = DmoroCardin
R = Matrix{Float64}(I, 2, 2)

# Transformations
time_transform = false
checkpoint = true
save_traj = false

# Do not modify below this line ----------------------------------------------
save_dir = "/home/dominic/JuliaProjects/LangevinIntegrators/outputs/$(exp_name)"

# Run the experiments
@info "Running: $(exp_name)"
master_2D_experiment(integrators, num_repeats, potential, diffusion, T, R, tau, stepsizes, xmin, ymin, xmax, ymax, n_bins, save_dir, chunk_size=1000000, checkpoint=false, q0=nothing, save_traj=false, time_transform=time_transform)
