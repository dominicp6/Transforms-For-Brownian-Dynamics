module FiniteTimeExperiments
include("../general_utils/calculus.jl")
include("../general_utils/potentials.jl")
include("../general_utils/diffusion_tensors.jl")
include("../general_utils/probability_utils.jl")
include("../general_utils/plotting_utils.jl")
include("../general_utils/transform_utils.jl")
include("../general_utils/misc_utils.jl")
include("experiments.jl")
using FHist, JLD2, Statistics, .Threads, ProgressBars, JSON, Random, StatsBase, Plots
import .Calculus: differentiate1D
import .ProbabilityUtils: compute_1D_mean_L1_error, compute_1D_invariant_distribution
import .PlottingUtils: save_and_plot
import .TransformUtils: increment_I_counts
import .MiscUtils: init_q0, create_directory_if_not_exists
import .DiffusionTensors: Dconst1D
import .Experiments: run_chunk
export run_1D_finite_time_convergence_experiment, finite_time_convergence_to_invariant_measure

"""
Compute the mean L1 error between a set of histograms and their corresponding reference histograms.

## Arguments
- `histograms`: An array of `Hist1D` objects representing the histograms to be compared with the reference histograms.
- `reference_histograms`: An array of `Hist1D` objects representing the reference histograms.

## Returns
- An array of floating-point values representing the mean L1 errors for each time snapshot.
"""
function compute_histogram_errors(histograms, reference_histograms)
    # Apply compute_1D_mean_L1_error for each corresponding pair of histograms
    errors = zeros(length(histograms))

    # Computing the error for each time snapshot
    for i in eachindex(histograms)
        errors[i] = compute_1D_mean_L1_error(histograms[i], reference_histograms[i])
    end

    return errors
end

function compute_histogram_errors_single_reference(histograms, reference_histogram, total_samples)
    # Apply compute_1D_mean_L1_error for each corresponding pair of histograms
    errors = zeros(length(histograms))

    # Computing the error for each time snapshot
    for i in eachindex(histograms)
        errors[i] = compute_1D_mean_L1_error(histograms[i], reference_histogram, total_samples)
    end

    return errors
end

"""
Plot finite-time errors for different integrators and stepsizes.

## Arguments
- `error`: A 3-dimensional array containing the error data for different integrators and stepsizes. The shape should be (num_integrators, num_stepsizes, num_snapshots), where num_integrators is the number of integrators, num_stepsizes is the number of stepsizes, and num_snapshots is the number of time snapshots.
- `integrators`: An array of integrator names.
- `stepsizes`: An array of stepsizes used in the simulation.
- `time_snapshots`: An array of time snapshots for which errors are computed.
- `save_dir`: The directory where the generated plots will be saved.
- `plot_type`: Optional. The type of plot to generate, defaults to "untransformed".
"""
function plot_finite_time_errors(error, integrators, stepsizes, time_snapshots, save_dir, plot_type="untransformed")
    # Plot the error data
    for (integrator_idx, integrator) in enumerate(integrators)
        for (stepsize_idx, stepsize) in enumerate(stepsizes)
            plot_title = "$(string(nameof(integrator))),$(round(stepsize,digits=3)),$(plot_type)"
            plot(time_snapshots, error[integrator_idx, stepsize_idx, :], xlabel="Time", ylabel="Error", title=plot_title)
            savefig("$(save_dir)/figures/h=$(round(stepsize,digits=3))/$(plot_type)/$(plot_title).png")
        end
    end

end

function plot_finite_time_errors_single_reference(error, integrator, stepsize, time_snapshots, save_dir)
    plot_title = "$(string(nameof(integrator))),$(round(stepsize,digits=3))"
    plot(time_snapshots, error, xlabel="Time", ylabel="Error", title=plot_title)
    savefig("$(save_dir)/figures/$(plot_title).png")

end

"""
Create necessary directories and save experiment parameters for the finite-time simulation experiment.

    ## Arguments
    - `save_dir`: The path to the directory where experiment data will be saved.
    - `integrators`: An array of integrator names for the simulation.
    - `reference_intgrator`: The name of the reference integrator.
    - `reference_stepsize`: The stepsize used for the reference integrator.
    - `untransformed`: A boolean indicating whether to create directories for untransformed histograms and figures.
    - `time_transform`: A boolean indicating whether to create directories for time-transformed histograms and figures.
    - `space_transform`: A boolean indicating whether to create directories for space-transformed histograms and figures.
    - `stepsizes`: An array of stepsizes to be used in the simulation.
    - `num_repeats`: The number of times to repeat the simulation.
    - `V`: The potential function used in the simulation.
    - `D`: The diffusion coefficient function used in the simulation.
    - `tau`: The noise strength parameter.
    - `x_bins`: The bin boundaries for the histograms.
    - `chunk_size`: The number of repeats to run in each computational chunk.
    - `ΔT`: The time interval for saving the distribution in the simulation.
    - `T`: The total simulation time.
"""
function create_experiment_folders(save_dir, integrators, reference_intgrator, reference_stepsize, untransformed::Bool, time_transform::Bool, space_transform::Bool,  stepsizes, num_repeats, V, D, tau, x_bins, chunk_size, ΔT, T)
    # Create master directory
    create_directory_if_not_exists(save_dir)
    
    # Create subdirectories for histograms
    create_directory_if_not_exists("$(save_dir)/histograms/reference")
    
    for integrator in integrators
        if untransformed
            create_directory_if_not_exists("$(save_dir)/histograms/$(nameof(integrator))/untransformed")
        end
        if time_transform
            create_directory_if_not_exists("$(save_dir)/histograms/$(nameof(integrator))/time_transformed")
        end
        if space_transform
            create_directory_if_not_exists("$(save_dir)/histograms/$(nameof(integrator))/space_transformed")
        end
    end
    
    # Create subdirectories for figures
    for stepsize in stepsizes
        if untransformed
            create_directory_if_not_exists("$(save_dir)/figures/h=$(round(stepsize,digits=3))/untransformed")
        end
        if time_transform
            create_directory_if_not_exists("$(save_dir)/figures/h=$(round(stepsize,digits=3))/time_transformed")
        end
        if space_transform
            create_directory_if_not_exists("$(save_dir)/figures/h=$(round(stepsize,digits=3))/space_transformed")
        end
    end
    
    # Create directory for results
    create_directory_if_not_exists("$(save_dir)/results")    

    if !isfile("$(save_dir)/experiment_params.json")
        # Save experiment parameters to file
        experiment_params = Dict("integrators" => [string(nameof(integrator)) for integrator in integrators],
                                "reference_integrator" => string(nameof(reference_intgrator)),
                                "reference_stepsize" => reference_stepsize,
                                "time_transform" => string(time_transform),
                                "space_transform" => string(space_transform),
                                "stepsizes" => stepsizes,
                                "num_repeats" => num_repeats,
                                "V" => string(nameof(V)),
                                "D" => string(nameof(D)),
                                "tau" => tau,
                                "x_bins" => x_bins,
                                "chunk_size" => chunk_size,
                                "ΔT" => ΔT,
                                "T" => T)
        open("$(save_dir)/experiment_params.json", "w") do f
            JSON.print(f, experiment_params, 4)
        end
    end
end

function run_1D_finite_time_experiment_untransformed(integrator, num_repeats, V, D, ΔT, T, tau, stepsize, bin_boundaries, save_dir; chunk_size=1000, mu0=nothing, sigma0=nothing, reference_simulation=false)
    
    # Set the mean and the variance of the ρ₀ distribution (distribution of starting configurations) if not specified 
    if (mu0 === nothing)
        mu0 = 0
    end

    if (sigma0 === nothing)
        sigma0 = 1
    end

    # Compute the symbolic derivatives of the potential and diffusion coefficients
    Vprime = differentiate1D(V)
    Dprime = differentiate1D(D)

    # Compute the number of time snapshots to save
    number_of_snapshots = Int(floor(T / ΔT))

    # Initialise an empty histogram for each time snapshot (to store the finite-time distributions)
    histograms = [Hist1D(0, bin_boundaries) for i in 1:number_of_snapshots]
    
    # While there are still trajectories to run
    repeats_remaining = num_repeats
    while repeats_remaining > 0

        # Compute the number of repeats to run in this computational chunk
        repeats_to_run = convert(Int, min(num_repeats, chunk_size))

        # Initialise an array to store the positions of the trajectories at each time snapshot
        snapshots = zeros(number_of_snapshots, repeats_to_run)

        Threads.@threads for repeat in ProgressBar(1:repeats_to_run)   
            
            # Initialise the starting position
            q0 = mu0 + sigma0 * randn()

            # Set the initial value of Rₖ to nothing
            previous_Rₖ = nothing

            # Simulate until T, saving the distribution every ΔT
            for snapshot_number in 1:number_of_snapshots

                # Compute the number of steps to run in this snapshot interval 
                steps_to_run = floor(Int, ΔT / stepsize)     

                # Run the chunk of steps
                q_chunk, Rₖ = integrator(q0, Vprime, D, Dprime, tau, steps_to_run, stepsize, previous_Rₖ)

                # Save the final position
                q0 = copy(q_chunk[end])

                # Update the previous value of Rₖ
                previous_Rₖ = Rₖ

                # Save the final snapshop position
                snapshots[snapshot_number, repeat] = q0
            end
        end

        repeats_remaining -= repeats_to_run

        # Construct histogram increments for each snapshot from the latest chunk of trajectories
        histogram_increments = [Hist1D(snapshots[i, :], bin_boundaries) for i in 1:number_of_snapshots]

        # Update the histograms
        for i in eachindex(histograms)
            histograms[i] += histogram_increments[i]
        end
    end

    # Save the histograms
    if reference_simulation
        save("$(save_dir)/histograms/reference/histograms.jld2", "data", histograms)
    else
        save("$(save_dir)/histograms/$(string(nameof(integrator)))/untransformed/histograms.jld2", "data", histograms)
    end

    return histograms
end

function run_1D_finite_time_experiment_time_transform(integrator, num_repeats, original_V, original_D, ΔT, T, tau, stepsize, bin_boundaries, save_dir; chunk_size=1000, checkpoint=false, save_traj=false, mu0=nothing, sigma0=nothing)
        
    # Set the mean and the variance of the ρ₀ distribution (distribution of starting configurations) if not specified 
    if (mu0 === nothing)
        mu0 = 0
    end

    if (sigma0 === nothing)
        sigma0 = 1
    end

    # Define the transformed potential and diffusion coefficients
    V = x -> original_V(x) - tau * log(original_D(x))
    D = Dconst1D

    # Compute the symbolic derivatives of the potential and diffusion coefficients
    Vprime = differentiate1D(V)
    Dprime = differentiate1D(D)

    # Compute the number of time snapshots to save
    number_of_snapshots = Int(floor(T / ΔT))

    # Initialise an empty histogram for each time snapshot (to store the finite-time distributions)
    histograms = [Hist1D(0, bin_boundaries) for i in 1:number_of_snapshots]
    
    repeats_remaining = num_repeats
    # While there are still trajectories to run...
    while repeats_remaining > 0

        # Compute the number of repeats to run in this computational chunk
        repeats_to_run = convert(Int, min(num_repeats, chunk_size))

        # Initialise an array to store the positions of the trajectories at each time snapshot
        snapshots = zeros(number_of_snapshots, repeats_to_run)

        Threads.@threads for repeat in ProgressBar(1:repeats_to_run)   
            
            # Initialise the starting position
            q0 = mu0 + sigma0 * randn()

            # Set the initial value of Rₖ to nothing
            previous_Rₖ = nothing

            # Initialise time remaining
            time_remaining = T

            # Initialise the time until the next snapshot
            time_until_next_snapshot = ΔT

            # Initialise the snapshot number
            snapshot_number = 0

            # Simulate until T, saving the position every ΔT
            while snapshot_number < number_of_snapshots
                # Compute the time rescaling factor g based on the current position
                g = 1/original_D(q0)

                # Run one step of the integrator 
                q1, Rₖ = integrator(q0, Vprime, D, Dprime, tau, 1, stepsize, previous_Rₖ)
                
                # Compute the elapsed time during this step (in the original time scale)
                delta_t = g * stepsize

                # Update the time remaining until next snapshot
                time_until_next_snapshot -= delta_t

                # If we have not yet reached the next snapshot
                if time_until_next_snapshot > 0
                    # Update the position and the previous value of Rₖ
                    previous_Rₖ = Rₖ
                    q0 = q1[1]
                else
                    # We have overshot, so we need to perform linear interpolation to estimate the position at the snapshot time
                    q_final = q0 + (time_until_next_snapshot / delta_t) * (q1[1] - q0)

                    # Update the snapshot number (if this reaches number_of_snapshots, the while loop will terminate after this iteration)
                    snapshot_number += 1
                    
                    # Save the final snapshot position
                    snapshots[snapshot_number, repeat] = q_final

                    # Compute the overshoot time (time_remaining_in_snapshot is negative)
                    overshoot_time = abs(time_until_next_snapshot)

                    # We dont resume from q_final to avoid accummulating bias from the linear interpolation made at each snapshot
                    # We resume our simulation from the sample just before overshooting (q0), so:
                    time_until_next_snapshot = ΔT + overshoot_time
                end
            end
        end

        #     # Simulate until T, saving the distribution every ΔT
        #     for snapshot_number in 1:number_of_snapshots

        #         time_remaining_in_snapshot = ΔT

        #         while time_remaining_in_snapshot > 0
        #             # Compute time rescaling factor (approximation - evaluate this at the initial position)
        #             g = 1/original_D(q0)
                    
        #             # Run one step of the integrator 
        #             q1, Rₖ = integrator(q0, Vprime, D, Dprime, tau, 1, stepsize, previous_Rₖ)

        #             # t time elapsed in this step
        #             delta_t = g * stepsize

        #             # Update the time remaining in this snapshot
        #             time_remaining_in_snapshot -= delta_t

        #             if time_remaining_in_snapshot > 0
        #                 # Update the previous value of Rₖ
        #                 previous_Rₖ = Rₖ

        #                 # Update the current position
        #                 q0 = q1[1]
        #             else
        #                 # Perform a linear interpolation to find the position at the end of the snapshot (approximation)
        #                 q0 = q0 + (q1[1] - q0) * (time_remaining_in_snapshot + delta_t) / delta_t
        #             end
        #         end 
    
        #         # Save the final snapshop position
        #         snapshots[snapshot_number, repeat] = q0
        #     end
        # end

        repeats_remaining -= repeats_to_run

        # Construct histogram increments for each snapshot from the latest chunk of trajectories
        histogram_increments = [Hist1D(snapshots[i, :], bin_boundaries) for i in 1:number_of_snapshots]

        # Update the histograms
        for i in eachindex(histograms)
            histograms[i] += histogram_increments[i]
        end
    end

    save("$(save_dir)/histograms/$(string(nameof(integrator)))/time_transformed/histograms.jld2", "data", histograms)

    return histograms
end

function run_1D_finite_time_experiment_space_transform(integrator, num_repeats, original_V, original_D, ΔT, T, tau, stepsize, bin_boundaries, save_dir; chunk_size=1000, checkpoint=false, save_traj=false, mu0=nothing, sigma0=nothing, x_of_y=nothing)

    # Set the mean and the variance of the ρ₀ distribution (distribution of starting configurations) if not specified 
    if (mu0 === nothing)
        mu0 = 0
    end

    if (sigma0 === nothing)
        sigma0 = 1
    end

    # Transform the potential so that the diffusion is constant
    @assert x_of_y !== nothing "x_of_y must be defined for space-transformed integrators"
    V = y -> original_V(x_of_y(y)) - 0.5 * tau * log(original_D(x_of_y(y)))
    D = Dconst1D

    # Compute the symbolic derivatives of the potential and diffusion coefficients
    Vprime = differentiate1D(V)
    Dprime = differentiate1D(D)  # Dprime is zero

    # Compute the number of time snapshots to save
    number_of_snapshots = Int(floor(T / ΔT))

    # Initialise an empty histogram for each time snapshot (to store the finite-time distributions)
    histograms = [Hist1D(0, bin_boundaries) for i in 1:number_of_snapshots]
    
    # While there are still trajectories to run
    repeats_remaining = num_repeats
    while repeats_remaining > 0

        # Compute the number of repeats to run in this computational chunk
        repeats_to_run = convert(Int, min(num_repeats, chunk_size))

        # Initialise an array to store the positions of the trajectories at each time snapshot
        snapshots = zeros(number_of_snapshots, repeats_to_run)

        Threads.@threads for repeat in ProgressBar(1:repeats_to_run)   
            
            # Initialise the starting position
            q0 = mu0 + sigma0 * randn()

            # Set the initial value of Rₖ to nothing
            previous_Rₖ = nothing

            # Simulate until T, saving the distribution every ΔT
            for snapshot_number in 1:number_of_snapshots

                # Compute the number of steps to run in this snapshot interval 
                steps_to_run = floor(Int, ΔT / stepsize)     

                # Run the chunk of steps
                q_chunk, Rₖ = integrator(q0, Vprime, D, Dprime, tau, steps_to_run, stepsize, previous_Rₖ)

                # Save the final position
                q0 = copy(q_chunk[end])

                # Update the previous value of Rₖ
                previous_Rₖ = Rₖ

                # Save the final snapshop position, transformed back to the original space
                snapshots[snapshot_number, repeat] = x_of_y(q0)
            end
        end

        repeats_remaining -= repeats_to_run

        # Construct histogram increments for each snapshot from the latest chunk of trajectories
        histogram_increments = [Hist1D(snapshots[i, :], bin_boundaries) for i in 1:number_of_snapshots]

        # Update the histograms
        for i in eachindex(histograms)
            histograms[i] += histogram_increments[i]
        end
    end

    # Save the histograms
    save("$(save_dir)/histograms/$(string(nameof(integrator)))/space_transformed/histograms.jld2", "data", histograms)

    return histograms
end

"""
Run a 1D finite-time convergence experiment for different integrators and stepsizes.

## Arguments
- `integrators`: An array of integrator names to be used in the untransformed experiments.
- `integrators_transformed`: An array of integrator names to be used in the transformed (space or time) experiments.
- `reference_integrator`: The name of the reference integrator.
- `reference_stepsize`: The stepsize used for the reference integrator.
- `num_repeats`: The number of times to repeat the simulation.
- `V`: The potential function used in the simulation.
- `D`: The diffusion coefficient function used in the simulation.
- `ΔT`: The time interval for saving the distribution in the simulation.
- `T`: The total simulation time.
- `tau`: The noise strength parameter for the simulation.
- `stepsizes`: An array of stepsizes to be used in the simulation.
- `bin_boundaries`: An array specifying the bin boundaries for histograms.
- `save_dir`: The path to the directory where experiment data and results will be saved.
- `chunk_size`: Optional. The number of repeats to run in each computational chunk, defaults to 1000.
- `mu0`: Optional. The mean of the starting position distribution, defaults to nothing.
- `sigma0`: Optional. The standard deviation of the starting position distribution, defaults to nothing.
- `untransformed`: Optional. A boolean indicating whether to run and analyze the untransformed simulation, defaults to true.
- `time_transform`: Optional. A boolean indicating whether to run and analyze the time-transformed simulation, defaults to false.
- `space_transform`: Optional. A boolean indicating whether to run and analyze the space-transformed simulation, defaults to false.
- `x_of_y`: Optional. A function representing the inverse transformation y = f(x) for space transformation, defaults to nothing.

## Description
The `run_1D_finite_time_convergence_experiment` function performs a 1D finite-time convergence experiment for different integrators and stepsizes. The experiment involves running the simulations and computing finite-time errors compared to a reference integrator. It also saves the results and generates plots for analysis.

The function accepts various optional parameters to control the experiment, such as starting position distribution (`mu0` and `sigma0`), transformations (`untransformed`, `time_transform`, and `space_transform`), and the inverse transformation function (`x_of_y`).

The experiment results are saved as JLD2 files, and plots are generated and saved based on the transformation options.
"""
function run_1D_finite_time_convergence_experiment(integrators, integrators_transformed, reference_integrator, reference_stepsize, num_repeats, V, D, ΔT, T, tau, stepsizes, bin_boundaries, save_dir; chunk_size=1000, mu0=nothing, sigma0=nothing, untransformed=true, time_transform=false, space_transform=false, x_of_y=nothing)

    @assert untransformed || time_transform || space_transform "At least one of untransformed, time_transform or space_transform must be set to true"

    # Create the experiment folders, if they don't already exist
    create_experiment_folders(save_dir, integrators, reference_integrator, reference_stepsize, untransformed, time_transform, space_transform, stepsizes, num_repeats, V, D, tau, bin_boundaries, chunk_size, ΔT, T)
    
    number_of_snapshots = Int(floor(T / ΔT))
    time_snapshots = ΔT:ΔT:T

    # Run a high accuracy simulation with a small stepsize 
    @info "Running reference simulation, $(string(nameof(reference_integrator))) with stepsize $(reference_stepsize) for $(T) seconds"
    reference_histograms = run_1D_finite_time_experiment_untransformed(reference_integrator, num_repeats, V, D, ΔT, T, tau, reference_stepsize, bin_boundaries, save_dir; chunk_size=chunk_size, mu0=mu0, sigma0=sigma0, reference_simulation=true)

    # Initialise error arrays
    if untransformed
        error_untransformed = zeros(length(integrators), length(stepsizes), number_of_snapshots)
    end
    if time_transform
        error_time_transformed = zeros(length(integrators_transformed), length(stepsizes), number_of_snapshots)
    end
    if space_transform
        error_space_transformed = zeros(length(integrators_transformed), length(stepsizes), number_of_snapshots)
    end  

    for (integrator_idx, integrator) in enumerate(integrators)

        # Repeat the experiment for the different specified step sizes 
        for (stepsize_idx, stepsize) in enumerate(stepsizes)

            # Check that ΔT is a multiple of stepsize
            tolerance = 1e-6  
            @assert (abs(ΔT % stepsize) < tolerance || abs(ΔT % stepsize - stepsize) < tolerance || abs(ΔT % stepsize + stepsize) < tolerance) "ΔT must be a multiple of stepsize"

            if untransformed
                @info "Running untransformed $(string(nameof(integrator))), stepsize = $stepsize"
                histograms = run_1D_finite_time_experiment_untransformed(integrator, num_repeats, V, D, ΔT, T, tau, stepsize, bin_boundaries, save_dir; chunk_size=chunk_size, mu0=mu0, sigma0=sigma0)
            
                # Compute the finite time error w.r.t the reference simulation for the different time snapshots 
                untransformed_finite_time_errors = compute_histogram_errors(histograms, reference_histograms)
                error_untransformed[integrator_idx, stepsize_idx, :] = untransformed_finite_time_errors
            end
        end
    end

    for (integrator_idx, integrator) in enumerate(integrators_transformed)

        for (stepsize_idx, stepsize) in enumerate(stepsizes)

            if time_transform
                @info "Running time transformed $(string(nameof(integrator))), stepsize = $stepsize"
                histograms_TT = run_1D_finite_time_experiment_time_transform(integrator, num_repeats, V, D, ΔT, T, tau, stepsize, bin_boundaries, save_dir; chunk_size=chunk_size, mu0=mu0, sigma0=sigma0)
                
                time_transformed_finite_time_errors = compute_histogram_errors(histograms_TT, reference_histograms)
                error_time_transformed[integrator_idx, stepsize_idx, :] = time_transformed_finite_time_errors
            end

            if space_transform
                @info "Running space transformed $(string(nameof(integrator))), stepsize = $stepsize"
                histograms_ST = run_1D_finite_time_experiment_space_transform(integrator, num_repeats, V, D, ΔT, T, tau, stepsize, bin_boundaries, save_dir; chunk_size=chunk_size, mu0=mu0, sigma0=sigma0, x_of_y=x_of_y)
                
                space_transformed_finite_time_errors = compute_histogram_errors(histograms_ST, reference_histograms)
                error_space_transformed[integrator_idx, stepsize_idx, :] = space_transformed_finite_time_errors
            end

        end

    end

    # Save the data to file
    if untransformed
        save("$(save_dir)/results/error.jld2", "data", error_untransformed)
    end
    if time_transform
        save("$(save_dir)/results/error_TT.jld2", "data", error_time_transformed)
    end
    if space_transform
        save("$(save_dir)/results/error_ST.jld2", "data", error_space_transformed)
    end

    # Plot finite time error results
    if untransformed
        plot_finite_time_errors(error_untransformed, integrators, stepsizes, time_snapshots, save_dir, "untransformed")
    end
    if time_transform
        plot_finite_time_errors(error_time_transformed, integrators_transformed, stepsizes, time_snapshots, save_dir, "time_transformed")
    end
    if space_transform
        plot_finite_time_errors(error_space_transformed, integrators_transformed, stepsizes, time_snapshots, save_dir, "space_transformed")
    end

end


function finite_time_convergence_to_invariant_measure(integrator, stepsize, num_repeats, V, D, ΔT, T, tau, bin_boundaries, save_dir; chunk_size=1000, mu0=0.0, sigma0=1.0)

    # Create the directory to save the results
    create_directory_if_not_exists(save_dir)
    create_directory_if_not_exists("$(save_dir)/figures")

    @info "Computing the Invariant Distribution"
    exact_invariant_distribution = compute_1D_invariant_distribution(V, tau, bin_boundaries)

    @info "Running finite time convergence experiment for $(string(nameof(integrator))) with stepsize $(stepsize) for $(T) seconds"
    histograms = run_1D_finite_time_experiment_untransformed(integrator, num_repeats, V, D, ΔT, T, tau, stepsize, bin_boundaries, save_dir; chunk_size=chunk_size, mu0=mu0, sigma0=sigma0)

    @info "Computing finite time errors"
    finite_time_errors = compute_histogram_errors_single_reference(histograms, exact_invariant_distribution, num_repeats)

    # Save the data to file
    save("$(save_dir)/finite_time_convergence.jld2", "data", finite_time_errors)

    # Plot finite time error results
    time_snapshots = ΔT:ΔT:T
    plot_finite_time_errors_single_reference(finite_time_errors, integrator, stepsize, time_snapshots, save_dir)

end

end # FiniteTimeExperiments