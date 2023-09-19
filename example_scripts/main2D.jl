include("general_utils/integrators.jl")
include("general_utils/potentials.jl")
include("general_utils/diffusion_tensors.jl")
include("general_utils/probability_utils.jl")
include("experiment_utils/experiments2D.jl")
using LinearAlgebra
import .Integrators: euler_maruyama2D, naive_leimkuhler_matthews2D, hummer_leimkuhler_matthews2D, stochastic_heun2D, leimkuhler_matthews2D    
import .Potentials: bowl2D, quadrupleWell2D, moroCardin2D, muller_brown, softQuadrupleWell2D
import .DiffusionTensors: Dconst2D, Dlinear2D, Dquadratic2D, DmoroCardin, Doseen
import .ProbabilityUtils:  compute_2D_invariant_distribution
import .Experiments2D: master_2D_experiment

"""
This script performs two-dimensional, variable-diffusion Brownian dynamics experiments and 
constructs plots of the weak convergence to the invariant measure for a range of specified step sizes.

Time rescalings are supported.
"""

# Name
exp_name = "2D_test"
master_dir = "path/to/results/directory" # Directory to save results in
T = 1000      # length of simulation
tau = 1       # noise coefficient
num_repeats = 12

# The step sizes to use (to use a single step size, set stepsizes = [stepsize])
num_step_sizes = 10
integrators = [leimkuhler_matthews2D]
stepsizes = 10 .^ range(-2.5,stop=-0.5,length=num_step_sizes)

# Histogram parameters for binning
xmin = -3
xmax = 3
ymin = -3
ymax = 3
n_bins = 30   # number of bins in each dimension

# The potential and diffusion coefficient to use
potential = softQuadrupleWell2D
diffusion = DmoroCardin
R = Matrix{Float64}(I, 2, 2)  # R matrix associated with the diffusion tensor (see paper for details)
                              # Here, the diffusion tensor is isotropic and so the R matrix is the identity matrix

# Information on the transformation
time_transform = false

# Whether to save checkpoints
checkpoint = true

# Do not modify below this line ----------------------------------------------
save_dir = "$(master_dir)/$(exp_name)"

# Run the experiments
@info "Running: $(exp_name)"
master_2D_experiment(integrators, num_repeats, potential, diffusion, T, R, tau, stepsizes, xmin, ymin, xmax, ymax, n_bins, save_dir, chunk_size=1000000, checkpoint=checkpoint, q0=nothing, time_transform=time_transform)
