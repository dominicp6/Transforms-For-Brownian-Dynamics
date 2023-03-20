include("integrators.jl")
include("plotting.jl")
include("potentials.jl")
include("diffusionTensors.jl")
include("utils.jl")
using Revise, Plots, JLD2, Statistics
import .Integrators: euler_maruyama, naive_leimkuhler_matthews, hummer_leimkuhler_matthews, euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D
import .Plotting: plot_trajectory, plot_2D_potential
import .Potentials: bowl2D, doubleWell1D, quadrupleWell2D, moroCardin2D, muller_brown, LM2013
import .DiffusionTensors: Dconst1D, Dlinear1D, Dquadratic1D, Dconst2D, Dlinear2D, Dquadratic2D, DmoroCardin, Doseen
import .Utils: compute_1D_probabilities, compute_2D_probabilities, compute_1D_mean_L1_error, compute_2D_mean_L1_error, run_1D_experiment, run_2D_experiment


# Temp parameter
tau = 1

# Time
T = 1000000

# Experiment parameters
num_repeats = 44
num_step_sizes = 10
stepsizes = 10 .^ range(-2.5,stop=-1,length=num_step_sizes)
integrators = [euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D]

# Histogram parameters
xmin = -3.5
ymin = -3.5
xmax = 3.5
ymax = 3.5
n_bins = 20

# Plot the potential
potential = LM2013
#display(plot_2D_potential(potential, -1.2, -1.2, 1.2, 1.2))

# Compute the expected counts in a 2D histogram of the configuration space
#probabilities, x_bins, y_bins, n_bins = compute_probabilty_grid(potential, tau, xmin, ymin, xmax, ymax, n_bins)
probabilities, x_bins, n_bins = compute_1D_probabilities(potential, tau, xmin, xmax, n_bins)

data = Dict()
# Run the experiments
for integrator in integrators
    convergence_data = run_1D_experiment(integrator, num_repeats, potential, Dconst1D, T, tau, stepsizes, probabilities, x_bins, n_bins, false)
    data[string(nameof(integrator))] = convergence_data
    println(convergence_data)
    save("convergence_data_$(integrator).jld2", "data", convergence_data)
end

# Plot the results
for integrator in integrators
    convergence_data = data[string(nameof(integrator))]
    # Plot the convergence data (dim 1 is the step size, dim 2 is the repeat)
    plot(stepsizes, mean(convergence_data, dims=2), title=string(nameof(integrator)), xlabel="dt", ylabel="Mean L1 error", xscale=:log10, yscale=:log10, label="")
    plot!(stepsizes, mean(convergence_data, dims=2)+std(convergence_data, dims=2), ls=:dash, lc=:black, label="")
    plot!(stepsizes, mean(convergence_data, dims=2)-std(convergence_data, dims=2), ls=:dash, lc=:black, label="")
    savefig("convergence_data_$(integrator).png")
end