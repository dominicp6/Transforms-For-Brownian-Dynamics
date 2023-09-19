module Experiments
include("../general_utils/calculus.jl")
include("../general_utils/potentials.jl")
include("../general_utils/diffusion_tensors.jl")
include("../general_utils/probability_utils.jl")
include("../general_utils/plotting_utils.jl")
include("../general_utils/misc_utils.jl")
include("../general_utils/transform_utils.jl")
using HCubature, QuadGK, FHist, JLD2, Statistics, .Threads, ProgressBars, JSON, Random, StatsBase, TimerOutputs, LombScargle, FFTW, Plots, Interpolations
import .Calculus: differentiate1D
import .ProbabilityUtils: compute_1D_mean_L1_error, compute_1D_invariant_distribution
import .PlottingUtils: save_and_plot
import .MiscUtils: init_q0, create_directory_if_not_exists
import .TransformUtils: increment_g_counts, increment_I_counts
import .DiffusionTensors: Dconst1D
export run_1D_experiment, master_1D_experiment, run_1D_experiment_until_given_error, run_autocorrelation_experiment, run_ess_autocorrelation_experiment

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
    q_chunk, _ = integrator(q0, Vprime, D, Dprime, tau, steps_to_run, dt)

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
- `tau`: The noise strength parameter.
- `stepsizes`: An array of step sizes to be used in the simulation.
- `probabilities`: The target probabilities to compute the convergence error.
- `bin_boundaries`: Bin boundaries for constructing histograms.
- `save_dir`: The directory path to save experiment results.
- `chunk_size`: Number of steps to run in each computational chunk to avoid memory issues.
- `checkpoint`: If true, save intermediate results in checkpoints.
- `q0`: The initial position of the trajectory. If not provided, it will be randomly initialized.
- `time_transform`: If true, apply time transformation to the potential and diffusion.
- `space_transform`: If true, apply space transformation to the potential and diffusion.
- `x_of_y`: A function that maps positions y to positions x for space-transformed integrators.

# Returns
- `convergence_errors`: A matrix containing convergence errors for each step size and repeat.

# Details
This function runs a 1D finite-time experiment with the specified integrator and system parameters. It supports various configurations, including time and space transformations. 
The experiment is repeated `num_repeats` times, each time with different initial conditions. For each combination of step size and repeat, the weak error w.r.t. the invariant distribution is computed.

Note: The `V` and `D` functions may be modified internally to implement time or space transformations, based on the provided `time_transform` and `space_transform` arguments.
"""
function run_1D_experiment(integrator, num_repeats, V, D, T, tau, stepsizes, probabilities, bin_boundaries, save_dir; chunk_size=10000000, checkpoint=false, q0=nothing, time_transform=false, space_transform=false, x_of_y=nothing)
    
    # Make master directory
    make_experiment_folders(save_dir, integrator, stepsizes, checkpoint, num_repeats, V, D, tau, bin_boundaries, chunk_size, time_transform, space_transform, T=T)

    original_D = D
    
    # [For transformed integrators] Modify the potential and diffusion functions appropriately (see paper for details)
    V, D = transform_potential_and_diffusion(V, D, tau, time_transform, space_transform, x_of_y)

    # Compute the symbolic derivative of the potential and diffusion functions
    Vprime = differentiate1D(V)
    Dprime = differentiate1D(D)

    # Initialise empty data array
    convergence_errors = zeros(length(stepsizes), num_repeats)

    for repeat in ProgressBar(1:num_repeats)
        # set the random seed for reproducibility
        Random.seed!(repeat) 

        # If no initial position is provided, randomly initialise
        q0 = init_q0(q0, dim=1)

        # Run the simulation for each specified step size
        for (stepsize_idx, dt) in enumerate(stepsizes)
            
            steps_remaining = floor(Int, T / dt)                
            total_samples = Int(steps_remaining)
            chunk_number = 0                                 
            hist = Hist1D([], bin_boundaries)            

            # For time-transformed integrators, we need to keep track of the following quantities for reweighting
            ΣgI = zeros(length(bin_boundaries)-1)
            Σg = 0.0

            # For space-transformed integrators, we need to keep track of the following quantities for reweighting
            ΣI = zeros(length(bin_boundaries)-1)

            while steps_remaining > 0
                # Run steps in chunks to minimise memory footprint
                steps_to_run = convert(Int, min(steps_remaining, chunk_size))
                q0, hist, chunk_number, ΣgI, Σg, ΣI = run_chunk(integrator, q0, Vprime, D, Dprime, tau, dt, steps_to_run, hist, bin_boundaries, chunk_number, time_transform, space_transform, ΣgI, Σg, ΣI, original_D, x_of_y)
                steps_remaining -= steps_to_run
            end

            # Compute the convergence error
            convergence_errors[stepsize_idx, repeat] = compute_convergence_error(hist, probabilities, total_samples, time_transform, space_transform, ΣgI, Σg, ΣI)

            if checkpoint
                # Save the histograms
                save("$(save_dir)/checkpoints/$(string(nameof(integrator)))/h=$dt/$(repeat).jld2", "data", hist)
            end
        end
    end

    # Save the error data and plot
    save_and_plot(integrator, convergence_errors, stepsizes, save_dir)

    @info "Mean L1 errors: $(mean(convergence_errors, dims=2))"
    @info "Standard deviation of L1 errors: $(std(convergence_errors, dims=2))"

    return convergence_errors
end

function transform_potential_and_diffusion(original_V, original_D, tau, time_transform, space_transform, x_of_y)
    @assert !(time_transform && space_transform) "Not supported to run both time and space transforms"
    
    if time_transform
        V = x -> original_V(x) - tau * log(original_D(x))
        D = Dconst1D
    end

    if space_transform
        @assert x_of_y !== nothing "x_of_y must be defined for space-transformed integrators"
        V = y -> original_V(x_of_y(y)) - 0.5 * tau * log(original_D(x_of_y(y)))
        D = Dconst1D
    end

    if !(time_transform || space_transform)
        V = original_V
        D = original_D
    end

    return V, D
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

"""
Run a 1D experiment using a given integrator until a specified error level (L1 error w.r.t. the exact invariant measure) is reached.

Parameters:
- `integrator`: The integrator to be used in the simulation.
- `num_repeats`: Number of experiment repeats to perform.
- `V`: Potential function V(x) representing the energy landscape.
- `D`: Diffusion function D(x) representing the diffusion coefficient.
- `tau`: The noise strength parameter.
- `stepsizes`: An array of time step sizes to be used in the simulation.
- `probabilities`: The target probability distribution to compare against.
- `bin_boundaries`: An array of bin boundaries for histogram computation.
- `save_dir`: The directory where results and checkpoints will be saved.
- `target_uncertainty`: The desired uncertainty level to be achieved.
- `chunk_size`: Number of simulation steps to be run in each chunk. Default is 10000.
- `checkpoint`: A boolean flag indicating whether to save checkpoints. Default is false.
- `q0`: The initial configuration for the simulation. Default is nothing, which generates a random configuration.
- `time_transform`: A boolean flag indicating whether to apply time transformation. Default is false.
- `space_transform`: A boolean flag indicating whether to apply space transformation. Default is false.
- `x_of_y`: A function that maps y-coordinates to corresponding x-coordinates for space-transformed integrators. Default is nothing.
    
Returns:
- `steps_until_uncertainty_data`: A matrix containing the number of steps taken until reaching the target uncertainty level for each step size and repeat.

Note:
- If `time_transform` is true, the potential function V(x) is transformed to ensure constant diffusion.
- If `space_transform` is true, the potential function V(x) is transformed based on the provided mapping x_of_y to ensure constant diffusion.
"""
function run_1D_experiment_until_given_error(integrator, num_repeats, V, D, tau, stepsizes, probabilities, bin_boundaries, save_dir, target_error; chunk_size=10000, checkpoint=false, q0=nothing, time_transform=false, space_transform=false, x_of_y=nothing)
    
    # Create the experiment folders
    make_experiment_folders(save_dir, integrator, stepsizes, checkpoint, num_repeats, V, D, tau, bin_boundaries, chunk_size, time_transform, space_transform, target_uncertainty=target_error)

    original_D = D

    # [For transformed integrators] Modify the potential and diffusion functions appropriately (see paper for details)
    V, D = transform_potential_and_diffusion(V, D, tau, time_transform, space_transform, x_of_y)

    # Compute the symbolic derivatives of the potential and diffusion functions
    Vprime = differentiate1D(V)
    Dprime = differentiate1D(D)

    # Initialise the data array
    steps_until_uncertainty_data = zeros(length(stepsizes), num_repeats)

    Threads.@threads for repeat in ProgressBar(1:num_repeats)
        # set the random seed for reproducibility
        Random.seed!(repeat) 
        q0 = init_q0(q0)

        # Run the simulation for each specified step size
        for (stepsize_idx, dt) in enumerate(stepsizes)

            steps_ran = 0                                    
            chunk_number = 0                                 
            error = Inf                                    
            hist = Hist1D([], bin_boundaries)                 

            # For time-transformed integrators, we need to keep track of the following quantities
            ΣgI = zeros(length(bin_boundaries)-1)
            Σg = 0.0

            # For space-transformed integrators, we need to keep track of the following quantities
            ΣI = zeros(length(bin_boundaries)-1)

            while error > target_error
                # Run steps in chunks to minimise memory footprint
                q0, hist, chunk_number, ΣgI, Σg, ΣI = run_chunk(integrator, q0, Vprime, D, Dprime, tau, dt, chunk_size, hist, bin_boundaries, chunk_number, time_transform, space_transform, ΣgI, Σg, ΣI, original_D, x_of_y)
                steps_ran += chunk_size
                
                # Compute the current error
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

            # Populate the data array
            if time_transform
                steps_until_uncertainty_data[stepsize_idx, repeat] = steps_ran
            elseif space_transform
                steps_until_uncertainty_data[stepsize_idx, repeat] = steps_ran
            else
                steps_until_uncertainty_data[stepsize_idx, repeat] = steps_ran
            end

            if checkpoint
                # Save the histograms
                save("$(save_dir)/checkpoints/$(string(nameof(integrator)))/h=$dt/$(repeat).jld2", "data", hist)
            end
        end
    end

    # Save the data and plot the results
    save_and_plot(integrator, steps_until_uncertainty_data, stepsizes, save_dir, ylabel="Steps until uncertainty < $(target_error)", error_in_mean=true)

    return steps_until_uncertainty_data
end

function run_ess_autocorrelation_experiment(integrator, num_repeats, V, D, T, tau, stepsize, max_lag, save_dir; time_transform=false, space_transform=false, x_of_y=nothing)
    @assert max_lag > 0
    @assert !(time_transform && space_transform)

    create_directory_if_not_exists(save_dir)

    # [For transformed integrators] Modify the potential and diffusion functions appropriately (see paper for details)
    V, D = transform_potential_and_diffusion(V, D, tau, time_transform, space_transform, x_of_y)

    # Compute the symbolic derivatives of the potential and diffusion functions
    Vprime = differentiate1D(V)
    Dprime = differentiate1D(D)

    num_samples = floor(Int, T/stepsize)

    max_lag = min(max_lag, num_samples)
    skip_first = 1 

    # Initialise empty arrays to store the autocorrelation data
    ac_data = zeros(max_lag, num_repeats)

    for repeat in ProgressBar(1:num_repeats)
        # randomise the random seed
        Random.seed!(time_ns())
        total_samples = floor(Int, T/stepsize)

        q0 = init_q0(nothing)

        q_traj, _ = integrator(q0, Vprime, D, Dprime, tau, total_samples, stepsize)

        # Compute the autocorrelation
        ac = autocor(q_traj, 1:max_lag)

        # Populate the data arrays
        ac_data[:, repeat] = ac
    end

    # Time series
    t = range(0, step=stepsize, length=max_lag)

    # Save the data and plot the results
    save_and_plot(integrator, ac_data[skip_first:end, :], t[skip_first:end], save_dir, xlabel="Time", ylabel="Autocorrelation", error_in_mean=true, descriptor="_untransformed", xscale=:identity, yscale=:identity)

    # Compute ESS
    mean_ac = mean(ac_data, dims=2)
    println(mean_ac)
    sum_ac = sum(mean_ac, dims=1)[1]
    println("Sum: $sum_ac")
    ess = 1/(1+ 2 * sum(mean_ac, dims=1)[1])
    println("ESS: $ess")

end

function run_autocorrelation_experiment(integrator, num_repeats, V, D, T, tau, stepsize, max_lag, save_dir; time_transform=false, space_transform=false, x_of_y=nothing)

    @assert max_lag > 0
    @assert !(time_transform && space_transform)

    create_directory_if_not_exists(save_dir)
    original_D = D

    # [For transformed integrators] Modify the potential and diffusion functions appropriately (see paper for details)
    V, D = transform_potential_and_diffusion(V, D, tau, time_transform, space_transform, x_of_y)

    # Compute the symbolic derivatives of the potential and diffusion functions
    Vprime = differentiate1D(V)
    Dprime = differentiate1D(D)

    num_samples = floor(Int, T/stepsize)

    max_lag = min(max_lag, num_samples)
    skip_first = 1 

    # Initialise empty arrays to store the autocorrelation data
    ac_sb_data = zeros(max_lag, num_repeats)

    for repeat in ProgressBar(1:num_repeats)
        # randomise the random seed
        Random.seed!(time_ns())
        total_samples = floor(Int, T/stepsize)

        q0 = init_q0(nothing)

        q_traj, _ = integrator(q0, Vprime, D, Dprime, tau, total_samples, stepsize)

        if space_transform
            # Transform the trajectory back to the original space before computing the autocorrelation
            q_traj = x_of_y.(q_traj)
        end

        if time_transform
            # Compute the t times associated with each sample
            t_times = zeros(total_samples)
            for i in 1:total_samples
                if i == 1
                    t_times[i] = 1/original_D(q_traj[i]) * stepsize
                else
                    t_times[i] = t_times[i-1] + 1/original_D(q_traj[i]) * stepsize
                end
            end

            # Compute an interpolated trajectory
            interp = linear_interpolation(t_times, q_traj)
            t_expected = stepsize:stepsize:maximum(t_times)
            q_traj = interp.(t_expected)
        end

        # Compute the autocorrelation
        ac_sb = autocor(q_traj, 1:max_lag)
        
        # Renormalise the autocorrelation
        ac_sb = ac_sb ./ ac_sb[1]

        # Populate the data arrays
        ac_sb_data[:, repeat] = ac_sb
    end

    # Time series
    t = range(0, step=stepsize, length=max_lag)

    # Save the data and plot the results
    save_and_plot(integrator, ac_sb_data[skip_first:end, :], t[skip_first:end], save_dir, xlabel="Time", ylabel="Autocorrelation", error_in_mean=true, descriptor="_SB", xscale=:identity, yscale=:identity)
end



"""
Run a master 1D experiment with multiple integrators.

Parameters:
- `integrators`: An array of integrators to be used in the experiments.
- `num_repeats`: Number of experiment repeats to perform for each integrator.
- `V`: Potential function V(x) representing the energy landscape.
- `D`: Diffusion function D(x) representing the diffusion coefficient.
- `T`: Total time for the simulation.
- `tau`: The noise strength parameter.
- `stepsizes`: An array of time step sizes to be used in the simulation.
- `bin_boundaries`: An array of bin boundaries for histogram computation.
- `save_dir`: The directory where results and time convergence data will be saved.
- `chunk_size`: Number of simulation steps to be run in each chunk. Default is 10000000.
- `checkpoint`: A boolean flag indicating whether to save checkpoints. Default is false.
- `q0`: The initial configuration for the simulation. Default is nothing, which generates a random configuration.
- `time_transform`: A boolean flag indicating whether to apply time transformation. Default is false.
- `space_transform`: A boolean flag indicating whether to apply space transformation. Default is false.
- `x_of_y`: A function that maps y-coordinates to corresponding x-coordinates for space-transformed integrators. Default is nothing.

Returns:
- The function saves the results of each experiment in the specified `save_dir` and also saves the time convergence data in a file named "time.json".
"""
function master_1D_experiment(integrators, num_repeats, V, D, T, tau, stepsizes, bin_boundaries, save_dir; chunk_size=10000000, checkpoint=false, q0=nothing, time_transform=false, space_transform=false, x_of_y=nothing)
    to = TimerOutput()

    @info "Computing the Invariant Distribution"
    exact_invariant_distribution = compute_1D_invariant_distribution(V, tau, bin_boundaries)

    @info "Running Experiments"
    for integrator in integrators
        @info "Running $(string(nameof(integrator))) experiment"
        @timeit to "Exp$(string(nameof(integrator)))" begin
            _ = run_1D_experiment(integrator, num_repeats, V, D, T, tau, stepsizes, exact_invariant_distribution, bin_boundaries, save_dir, chunk_size=chunk_size, checkpoint=checkpoint, q0=q0, time_transform=time_transform, space_transform=space_transform, x_of_y=x_of_y)
        end
    end

    # save the time convergence_data
    open("$(save_dir)/time.json", "w") do io
        JSON.print(io, TimerOutputs.todict(to), 4)
    end
end

end # module