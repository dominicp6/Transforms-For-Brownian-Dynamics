module Experiments
include("../general_utils/calculus.jl")
include("../general_utils/potentials.jl")
include("../general_utils/diffusion_tensors.jl")
include("../general_utils/probability_utils.jl")
include("../general_utils/plotting_utils.jl")
include("../general_utils/misc_utils.jl")
include("../general_utils/transform_utils.jl")
include("../general_utils/dynamics_utils.jl")
using HCubature, QuadGK, FHist, JLD2, Statistics, .Threads, ProgressBars, JSON, Random, StatsBase, TimerOutputs
import .Calculus: differentiate1D
import .ProbabilityUtils: compute_1D_mean_L1_error, compute_1D_probabilities
import .PlottingUtils: save_and_plot
import .MiscUtils: init_q0, create_directory_if_not_exists
import .TransformUtils: increment_g_counts, increment_I_counts
import .DynamicsUtils: run_estimate_diffusion_coefficient, run_estimate_diffusion_coefficient_time_rescaling, run_estimate_diffusion_coefficient_lamperti
import .DiffusionTensors: Dconst1D
export run_1D_experiment, master_1D_experiment, run_1D_experiment_until_given_uncertainty

"""
Creates necessary directories and save experiment parameters for the 1D experiment.
"""
function make_experiment_folders(save_dir, integrator, stepsizes, checkpoint, num_repeats, V, D, tau, x_bins, chunk_size, time_transform, space_transform; T=nothing, target_uncertainty=nothing)
    # Make master directory
    create_directory_if_not_exists(save_dir)

    # Make subdirectories for checkpoints
    if checkpoint
        create_directory_if_not_exists("$(save_dir)/checkpoints")
        for dt in stepsizes
            create_directory_if_not_exists("$(save_dir)/checkpoints/$(string(nameof(integrator)))/h=$dt")
        end
    end

    if !isfile("$(save_dir)/info.json")
        @info "Saving metadata"
        metadata = Dict("integrator" => string(nameof(integrator)),
                        "num_repeats" => num_repeats,
                        "V" => string(nameof(V)),
                        "D" => string(nameof(D)),
                        "T" => T,
                        "target_uncertainty" => target_uncertainty,
                        "tau" => tau,
                        "stepsizes" => stepsizes,
                        "x_bins" => x_bins,
                        "chunk_size" => chunk_size,
                        "time_transform" => string(time_transform),
                        "space_transform" => string(space_transform))
        open("$(save_dir)/info.json", "w") do f
            JSON.print(f, metadata, 4)
        end
    end
end

"""
The `run_chunk` function runs a chunk of the 1D finite-time simulation using the specified integrator and parameters.
It performs the simulation for `steps_to_run` time steps and updates the histogram with the trajectory data.

Note: The function is typically called within the context of the main simulation loop, and its results are used for further analysis.
"""
function run_chunk(integrator, q0, Vprime, D, Dprime, tau::Number, dt::Number, steps_to_run::Integer, hist, bin_boundaries, chunk_number::Integer, time_transform::Bool, space_transform:: Bool, ΣgI::Union{Vector, Nothing}, Σg::Union{Float64, Nothing}, ΣI::Union{Vector, Nothing}, original_D, x_of_y)

    # Run a chunk of the simulation
    q_chunk = integrator(q0, Vprime, D, Dprime, tau, steps_to_run, dt)

    # Get the last position of the chunk
    q0 = copy(q_chunk[end])

    # [For time-transformed integrators] Increment g counts (see paper for details)
    if time_transform
        ΣgI, Σg =  increment_g_counts(q_chunk, original_D, bin_boundaries, ΣgI, Σg)
    end

    # [For space-transformed integrators] Increment I counts (see paper for details)
    if space_transform
        ΣI = increment_I_counts(q_chunk, x_of_y, bin_boundaries, ΣI)
    end

    # Update the number of steps left to run
    hist += Hist1D(q_chunk, bin_boundaries)
    chunk_number += 1

    return q0, hist, chunk_number, ΣgI, Σg, ΣI
end

"""
Run a 1D finite-time experiment using the specified integrator and parameters.

# Arguments
- `integrator`: The integrator function to use for the simulation.
- `num_repeats`: Number of repeats for the experiment.
- `V`: The potential function that describes the energy landscape.
- `D`: The diffusion coefficient function that defines the noise in the system.
- `T`: Total simulation time.
- `tau`: The time step for the integrator.
- `stepsizes`: An array of step sizes to be used in the simulation.
- `probabilities`: The target probabilities to compute the convergence error.
- `bin_boundaries`: Bin boundaries for constructing histograms.
- `save_dir`: The directory path to save experiment results.
- `chunk_size`: Number of steps to run in each computational chunk to avoid memory issues.
- `checkpoint`: If true, save intermediate results in checkpoints.
- `q0`: The initial position of the trajectory. If not provided, it will be randomly initialized.
- `save_traj`: If true, save trajectory data.
- `time_transform`: If true, apply time transformation to the potential and diffusion.
- `space_transform`: If true, apply space transformation to the potential and diffusion.
- `x_of_y`: A function that maps positions y to positions x for space-transformed integrators.

# Returns
- `convergence_data`: A matrix containing convergence errors for each step size and repeat.

# Details
This function runs a 1D finite-time experiment with the specified integrator and system parameters. It supports various configurations, including time and space transformations. 
The experiment is repeated `num_repeats` times, each time with different initial conditions. For each combination of step size and repeat, the weak error w.r.t. the invariant distribution is computed.

Note: The `V` and `D` functions may be modified internally to implement time or space transformations, based on the provided `time_transform` and `space_transform` arguments.
"""
function run_1D_experiment(integrator, num_repeats, V, D, T, tau, stepsizes, probabilities, bin_boundaries, save_dir; chunk_size=10000000, checkpoint=false, q0=nothing, time_transform=false, space_transform=false, x_of_y=nothing)
    make_experiment_folders(save_dir, integrator, stepsizes, checkpoint, num_repeats, V, D, tau, bin_boundaries, chunk_size, time_transform, space_transform, T=T)

    original_D = D
    
    transform_potential_and_diffusion!(V, D, time_transform, space_transform, tau, x_of_y)

    Vprime = differentiate1D(V)
    Dprime = differentiate1D(D)

    @info "Running $(string(nameof(integrator))) experiment with $num_repeats repeats"
    convergence_data = zeros(length(stepsizes), num_repeats)

    Threads.@threads for repeat in ProgressBar(1:num_repeats)
        Random.seed!(repeat) # set the random seed for reproducibility
        q0 = init_q0(q0, dim=1)

        for (stepsize_idx, dt) in enumerate(stepsizes)
            steps_remaining = floor(T / dt)                
            total_samples = Int(steps_remaining)
            chunk_number = 0                                 
            hist = Hist1D([], bin_boundaries)            

            # For time-transformed integrators
            ΣgI = zeros(length(bin_boundaries)-1)
            Σg = 0.0

            # For space-transformed integrators
            ΣI = zeros(length(bin_boundaries)-1)

            # Run steps in chunks to avoid memory issues
            while steps_remaining > 0
                steps_to_run = convert(Int, min(steps_remaining, chunk_size))
                q0, hist, chunk_number, ΣgI, Σg, ΣI = run_chunk(integrator, q0, Vprime, D, Dprime, tau, dt, steps_to_run, hist, bin_boundaries, chunk_number, time_transform, space_transform, ΣgI, Σg, ΣI, original_D, x_of_y)
                steps_remaining -= steps_to_run
            end

            convergence_data[stepsize_idx, repeat] = compute_convergence_error(hist, probabilities, total_samples, time_transform, space_transform, ΣgI, Σg, ΣI)

            if checkpoint
                save("$(save_dir)/checkpoints/$(string(nameof(integrator)))/h=$dt/$(repeat).jld2", "data", hist)
            end
        end
    end

    save_and_plot(integrator, convergence_data, stepsizes, save_dir)

    @info "Mean L1 errors: $(mean(convergence_data, dims=2))"
    @info "Standard deviation of L1 errors: $(std(convergence_data, dims=2))"

    return convergence_data
end

function transform_potential_and_diffusion!(V, D, tau, time_transform, space_transform, x_of_y)
    @assert !(time_transform && space_transform) "Not supported to run both time and space transforms"
    
    if time_transform
        V = x -> V(x) - tau * log(D(x))
        D = Dconst1D
    end

    if space_transform
        @assert x_of_y !== nothing "x_of_y must be defined for space-transformed integrators"
        V = y -> V(x_of_y(y)) - 0.5 * tau * log(D(x_of_y(y)))
        D = Dconst1D
    end
end

function compute_convergence_error(hist, probabilities, total_samples, time_transform, space_transform, ΣgI, Σg, ΣI)
    if time_transform
        empirical_probabilities = ΣgI ./ Σg
        return compute_1D_mean_L1_error(empirical_probabilities, probabilities)
    elseif space_transform
        empirical_probabilities = ΣI ./ total_samples
        return compute_1D_mean_L1_error(empirical_probabilities, probabilities)
    else
        return compute_1D_mean_L1_error(hist, probabilities, total_samples)
    end
end

function run_1D_experiment_until_given_uncertainty(integrator, num_repeats, V, D, tau, stepsizes, probabilities, bin_boundaries, save_dir, target_uncertainty; chunk_size=10000, checkpoint=false, q0=nothing, time_transform=false, space_transform=false, x_of_y=nothing)
    make_experiment_folders(save_dir, integrator, stepsizes, checkpoint, num_repeats, V, D, tau, bin_boundaries, chunk_size, time_transform, space_transform, target_uncertainty=target_uncertainty)

    original_D = D
    original_V = V

    @assert !(time_transform && space_transform) "Invalid to run both time and space transforms"

    if time_transform
        # Transform the potential so that the diffusion is constant
        V = x -> original_V(x) - tau * log(original_D(x))
        D = Dconst1D
    end

    if space_transform
        # Transform the potential so that the diffusion is constant
        @assert x_of_y !== nothing "x_of_y must be defined for space-transformed integrators"
        V = y -> original_V(x_of_y(y)) - 0.5 * tau * log(original_D(x_of_y(y)))
        D = Dconst1D
    end

    Vprime = differentiate1D(V)
    Dprime = differentiate1D(D)

    @info "Running $(string(nameof(integrator))) experiment with $num_repeats repeats"
    steps_until_uncertainty_data = zeros(length(stepsizes), num_repeats)

    Threads.@threads for repeat in ProgressBar(1:num_repeats)

        Random.seed!(repeat) # set the random seed for reproducibility
        q0 = init_q0(q0)

        for (stepsize_idx, dt) in enumerate(stepsizes)

            # Initialise for this step size
            steps_ran = 0                                     # total number of steps
            chunk_number = 0                                  # number of chunks run so far
            error = Inf                                       # error in the histogram
            hist = Hist1D([], bin_boundaries)                 # histogram of the trajectory

            # For time-transformed integrators
            ΣgI = zeros(length(bin_boundaries)-1)
            Σg = 0.0

            # For space-transformed integrators
            ΣI = zeros(length(bin_boundaries)-1)

            # Run steps in chunks to avoid memory issues
            while error > target_uncertainty
                q0, hist, chunk_number, ΣgI, Σg, ΣI = run_chunk(integrator, q0, Vprime, D, Dprime, tau, dt, chunk_size, hist, bin_boundaries, save_dir, repeat, chunk_number, save_traj, time_transform, space_transform, ΣgI, Σg, ΣI, original_D, x_of_y)
                steps_ran += chunk_size
                if time_transform
                    empirical_probabilities = ΣgI ./ Σg
                    error = compute_1D_mean_L1_error(empirical_probabilities, probabilities)
                elseif space_transform
                    empirical_probabilities = ΣI ./ steps_ran
                    error = compute_1D_mean_L1_error(empirical_probabilities, probabilities)
                else
                    error = compute_1D_mean_L1_error(hist, probabilities, steps_ran)
                end
            end

            if time_transform
                empirical_probabilities = ΣgI ./ Σg
                steps_until_uncertainty_data[stepsize_idx, repeat] = steps_ran
            elseif space_transform
                empirical_probabilities = ΣI ./ steps_ran
                steps_until_uncertainty_data[stepsize_idx, repeat] = steps_ran
            else
                steps_until_uncertainty_data[stepsize_idx, repeat] = steps_ran
            end

            if checkpoint
                save("$(save_dir)/checkpoints/$(string(nameof(integrator)))/h=$dt/$(repeat).jld2", "data", hist)
            end
        end
    end

    save_and_plot(integrator, steps_until_uncertainty_data, stepsizes, save_dir, ylabel="Steps until uncertainty < $(target_uncertainty)", error_in_mean=true)

    return steps_until_uncertainty_data
end


function master_1D_experiment(integrators, num_repeats, V, D, T, tau, stepsizes, bin_boundaries, save_dir; chunk_size=10000000, checkpoint=false, q0=nothing, save_traj=false, time_transform=false, space_transform=false, x_of_y=nothing)
    to = TimerOutput()

    @info "Computing Expected Probabilities"
    probabilities = compute_1D_probabilities(V, tau, bin_boundaries)

    @info "Running Experiments"
    for integrator in integrators
        @info "Running $(string(nameof(integrator))) experiment"
        @timeit to "Exp$(string(nameof(integrator)))" begin
            convergence_data = run_1D_experiment(integrator, num_repeats, V, D, T, tau, stepsizes, probabilities, bin_boundaries, save_dir, chunk_size=chunk_size, checkpoint=checkpoint, q0=q0, save_traj=save_traj, time_transform=time_transform, space_transform=space_transform, x_of_y=x_of_y)
        end
    end

    # save the time convergence_data
    open("$(save_dir)/time.json", "w") do io
        JSON.print(io, TimerOutputs.todict(to), 4)
    end
end

end # module