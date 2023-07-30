module FiniteTimeExperiments
include("../general_utils/calculus.jl")
include("../general_utils/potentials.jl")
include("../general_utils/diffusion_tensors.jl")
include("../general_utils/probability_utils.jl")
include("../general_utils/plotting_utils.jl")
include("../general_utils/misc_utils.jl")
include("../general_utils/dynamics_utils.jl")
include("experiments.jl")
using FHist, JLD2, Statistics, .Threads, ProgressBars, JSON, Random, StatsBase, Plots
import .Calculus: differentiate1D
import .ProbabilityUtils: compute_1D_mean_L1_error, compute_1D_probabilities
import .PlottingUtils: save_and_plot
import .MiscUtils: init_q0
import .DynamicsUtils: run_estimate_diffusion_coefficient, run_estimate_diffusion_coefficient_time_rescaling, run_estimate_diffusion_coefficient_lamperti
import .DiffusionTensors: Dconst1D
import .Experiments: run_chunk
export run_1D_finite_time_convergence_experiment

function compute_histogram_errors(histograms, reference_histograms)
    # Apply compute_1D_mean_L1_error for each corresponding pair of histograms
    errors = zeros(length(histograms))

    # Computing the error for each time snapshot
    for i in eachindex(histograms)
        errors[i] = compute_1D_mean_L1_error(histograms[i], reference_histograms[i])
    end

    return errors
end

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

function create_experiment_folders(save_dir, integrators, reference_intgrator, reference_stepsize, time_transform::Bool, space_transform::Bool,  stepsizes, num_repeats, V, D, tau, x_bins, chunk_size, ΔT, T)
    function create_directory_if_not_exists(dir_path)
        if !isdir(dir_path)
            mkpath(dir_path)
            @info "Created directory $dir_path"
        end
    end
    
    # Create master directory
    create_directory_if_not_exists(save_dir)
    
    # Create subdirectories for histograms
    create_directory_if_not_exists("$(save_dir)/histograms/reference")
    
    for integrator in integrators
        create_directory_if_not_exists("$(save_dir)/histograms/$(nameof(integrator))/untransformed")
        if time_transform
            create_directory_if_not_exists("$(save_dir)/histograms/$(nameof(integrator))/time_transformed")
        end
        if space_transform
            create_directory_if_not_exists("$(save_dir)/histograms/$(nameof(integrator))/space_transformed")
        end
    end
    
    # Create subdirectories for figures
    for stepsize in stepsizes
        create_directory_if_not_exists("$(save_dir)/figures/h=$(round(stepsize,digits=3))/untransformed")
        if time_transform
            create_directory_if_not_exists("$(save_dir)/figures/h=$(round(stepsize,digits=3))/time_transformed")
        end
        if space_transform
            create_directory_if_not_exists("$(save_dir)/figures/h=$(round(stepsize,digits=3))/space_transformed")
        end
    end
    
    # Create directory for results
    create_directory_if_not_exists("$(save_dir)/results")    

    # If space-transformed, create appropriate subdirectories
    if space_transform
        for integrator in integrators
            if !isdir("$(save_dir)/histograms/$(string(nameof(integrator)))/space_transformed")
                mkdir("$(save_dir)/histograms/$(string(nameof(integrator)))/space_transformed")
                @info "Created directory $(save_dir)/histograms/$(string(nameof(integrator)))/space_transformed"
            end
        end

        for stepsize in stepsizes
            if !isdir("$(save_dir)/figures/h=$(round(stepsize,digits=3))/space_transformed")
                mkdir("$(save_dir)/figures/h=$(round(stepsize,digits=3))/space_transformed")
                @info "Created directory $(save_dir)/figures/h=$(round(stepsize,digits=3))/space_transformed"
            end
        end
    end

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

function run_1D_finite_time_experiment(integrator, num_repeats, V, D, ΔT, T, tau, stepsize, bin_boundaries, save_dir; chunk_size=1000, save_traj=false, mu0=nothing, sigma0=nothing, reference_simulation=false)
    # ΔT is the time between saving distribution snapshots

    # Set mean and variance of ρ₀ distribution if not specified 
    if (mu0 === nothing)
        mu0 = 0
    end

    if (sigma0 === nothing)
        sigma0 = 1
    end

    # Compute Vprime and Dprime
    Vprime = differentiate1D(V)
    Dprime = differentiate1D(D)

    # Run the experiment
    @info "Running $(string(nameof(integrator))) experiment with $num_repeats repeats"

    # Initialise q0 final values
    number_of_snapshots = Int(floor(T / ΔT))

    repeats_remaining = num_repeats
    histograms = [Hist1D(0, bin_boundaries) for i in 1:number_of_snapshots]
    while repeats_remaining > 0
        repeats_to_run = convert(Int, min(num_repeats, chunk_size))
        q0_finals = zeros(number_of_snapshots, repeats_to_run)

        Threads.@threads for repeat in ProgressBar(1:repeats_to_run)
            hist = Hist1D([], bin_boundaries)                     # histogram of the trajectory
            q0 = mu0 + sigma0 * randn()

            # Simulate until T, saving the distribution every ΔT
            for snapshot_number in 1:number_of_snapshots
                steps_remaining = floor(ΔT / stepsize)            # total number of steps
                chunk_number = 0                                  # number of chunks run so far

                while steps_remaining > 0
                    steps_to_run = convert(Int, min(steps_remaining, chunk_size))
                    q0, hist, chunk_number, _, _, _, _, _ = run_chunk(integrator, q0, Vprime, D, Dprime, tau, stepsize, steps_to_run, hist, bin_boundaries, save_dir, repeat, chunk_number, save_traj, false, false, nothing, nothing, nothing, nothing, nothing)
                    steps_remaining -= steps_to_run
                end

                # Save the q0 final values
                q0_finals[snapshot_number, repeat] = q0
            end
        end

        repeats_remaining -= repeats_to_run

        # Construct new histograms
        new_histograms = [Hist1D(q0_finals[i, :], bin_boundaries) for i in 1:number_of_snapshots]

        # Add new histograms to existing histograms
        for i in eachindex(histograms)
            histograms[i] += new_histograms[i]
        end
    end

    # Plot the histograms
    # for (i, hist) in enumerate(histograms)
    #     plot_title = "$(string(nameof(integrator))),$(round(stepsize,digits=3)),$(i)"
    #     println("bincenters(hist) = $(bincenters(hist))")
    #     println("bincounts(hist) = $(bincounts(hist))")
    #     display(plot(bincenters(hist), bincounts(hist), xlabel="x", ylabel="y", title=plot_title))
    #     println("Press any key to continue:")
    #     readline()
    # end

    # Save the histograms
    if reference_simulation
        save("$(save_dir)/histograms/reference/histograms.jld2", "data", histograms)
    else
        save("$(save_dir)/histograms/$(string(nameof(integrator)))/untransformed/histograms.jld2", "data", histograms)
    end

    return histograms
end

function run_1D_finite_time_experiment_time_transform(integrator, num_repeats, original_V, original_D, ΔT, T, tau, stepsize, bin_boundaries, save_dir; chunk_size=1000, checkpoint=false, save_traj=false, mu0=nothing, sigma0=nothing)
    # ΔT is the time between saving distribution snapshots

    # Set mean and variance of ρ₀ distribution if not specified 
    if (mu0 === nothing)
        mu0 = 0
    end

    if (sigma0 === nothing)
        sigma0 = 1
    end
    
    # Transform the potential so that the diffusion is constant
    V = x -> original_V(x) - tau * log(original_D(x))
    D = Dconst1D

    # Compute Vprime and Dprime
    Vprime = differentiate1D(V)
    Dprime = differentiate1D(D)

    # Run a single chunk to estimate the time rescaling factor
    q0 = mu0 + sigma0 * randn()
    ΣgI = zeros(length(bin_boundaries)-1)            
    Σg = 0.0
    hist = Hist1D([], bin_boundaries) 
    chunk_number = 0
    steps_to_run = 1000
    repeat = 0
    q0, hist, chunk_number, ΣgI, Σg, _, _, _ = run_chunk(integrator, q0, Vprime, D, Dprime, tau, stepsize, steps_to_run, hist, bin_boundaries, save_dir, repeat, chunk_number, save_traj, true, false, ΣgI, Σg, nothing, original_D, nothing)
    TRS = Σg / steps_to_run

    # Run the experiment
    @info "Running $(string(nameof(integrator))) experiment with $num_repeats repeats"

    # Initialise q0 final values
    number_of_snapshots = Int(floor(T / ΔT))
    repeats_remaining = num_repeats
    histograms = [Hist1D(0, bin_boundaries) for i in 1:number_of_snapshots]
    while repeats_remaining > 0
        repeats_to_run = convert(Int, min(num_repeats, chunk_size))
        q0_finals = zeros(number_of_snapshots, repeats_to_run)

        Threads.@threads for repeat in ProgressBar(1:repeats_to_run)
            hist = Hist1D([], bin_boundaries)            # histogram of the trajectory   

            q0 = mu0 + sigma0 * randn()

            # Simulate until T, saving the distribution every ΔT
            for snapshot_number in 1:number_of_snapshots
                time_remaining = ΔT                      # total time remaining for this snapshot
                chunk_number = 0                         # number of chunks run so far
                
                # For time-transformed integrators
                ΣgI = zeros(length(bin_boundaries)-1)            
                Σg = 0.0
                previous_Σg = Σg
                previous_q0 = q0

                while time_remaining > 0
                    previous_Σg = Σg
                    previous_q0 = q0
                    # Run 80% of the estimated steps remaining, plus one extra to allow for overshoot
                    estimated_steps_remaining = Int(floor(0.5 * time_remaining / (stepsize * TRS)) + 1)
                    steps_to_run = convert(Int, min(estimated_steps_remaining, chunk_size))
                    q0, hist, chunk_number, ΣgI, Σg, _, _, _ = run_chunk(integrator, q0, Vprime, D, Dprime, tau, stepsize, steps_to_run, hist, bin_boundaries, save_dir, repeat, chunk_number, save_traj, true, false, ΣgI, Σg, nothing, original_D, nothing)
                    delta_t = (Σg - previous_Σg) * stepsize
                    time_remaining -= delta_t
                end

                # Run steps individually until overshoot 
                steps_to_run = 1              
                Σg = previous_Σg
                q0 = previous_q0
                time_remaining += delta_t
                delta_t = 0.0
                while time_remaining > 0
                    previous_Σg = Σg
                    previous_q0 = q0
                    q0, hist, chunk_number, ΣgI, Σg, _, _, _ = run_chunk(integrator, q0, Vprime, D, Dprime, tau, stepsize, steps_to_run, hist, bin_boundaries, save_dir, repeat, chunk_number, save_traj, true, false, ΣgI, Σg, nothing, original_D, nothing)
                    delta_t = (Σg - previous_Σg) * stepsize
                    time_remaining -= delta_t
                end

                # Once the time is overshot, run a partial step to reach the correct time
                partial_stepsize = stepsize * (1 + time_remaining / delta_t)
                q0, _, _, _, _, _, _, _ = run_chunk(integrator, previous_q0, Vprime, D, Dprime, tau, partial_stepsize, 1, hist, bin_boundaries, save_dir, repeat, chunk_number, save_traj, true, false, ΣgI, Σg, nothing, original_D, nothing)

                # Save the q0 final values
                q0_finals[snapshot_number, repeat] = q0
            end
        end

        repeats_remaining -= repeats_to_run

        # Construct new histograms
        new_histograms = [Hist1D(q0_finals[i, :], bin_boundaries) for i in 1:number_of_snapshots]

        # Add new histograms to existing histograms
        for i in eachindex(histograms)
            histograms[i] += new_histograms[i]
        end

    end

    # # Plot the histograms
    # for (i, hist) in enumerate(histograms)
    #     plot_title = "$(string(nameof(integrator))),$(round(stepsize,digits=3)),$(i)"
    #     println("bincenters(hist) = $(bincenters(hist))")
    #     println("bincounts(hist) = $(bincounts(hist))")
    #     display(plot(bincenters(hist), bincounts(hist), xlabel="x", ylabel="y", title=plot_title))
    #     println("Press any key to continue:")
    #     readline()
    # end

    # Save the histograms
    save("$(save_dir)/histograms/$(string(nameof(integrator)))/time_transformed/histograms.jld2", "data", histograms)

    return histograms
end

function run_1D_finite_time_experiment_space_transform(integrator, num_repeats, original_V, original_D, ΔT, T, tau, stepsize, bin_boundaries, save_dir; chunk_size=1000, checkpoint=false, save_traj=false, mu0=nothing, sigma0=nothing, x_of_y=nothing)
    # ΔT is the time between saving distribution snapshots

    # Set mean and variance of ρ₀ distribution if not specified 
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

    # Compute Vprime and Dprime
    Vprime = differentiate1D(V)
    Dprime = differentiate1D(D)

    # Run the experiment
    @info "Running $(string(nameof(integrator))) experiment with $num_repeats repeats"

    # Initialise q0 final values
    number_of_snapshots = Int(floor(T / ΔT))
    repeats_remaining = num_repeats
    histograms = [Hist1D(0, bin_boundaries) for i in 1:number_of_snapshots]
    while repeats_remaining > 0
        repeats_to_run = convert(Int, min(num_repeats, chunk_size))
        q0_finals = zeros(number_of_snapshots, repeats_to_run)

        Threads.@threads for repeat in ProgressBar(1:repeats_to_run)
            hist = Hist1D([], bin_boundaries)                     # histogram of the trajectory
            q0 = mu0 + sigma0 * randn()

            # Simulate until T, saving the distribution every ΔT
            for snapshot_number in 1:number_of_snapshots
                steps_remaining = floor(ΔT / stepsize)            # total number of steps
                chunk_number = 0                                  # number of chunks run so far
                ΣI = zeros(length(bin_boundaries)-1)           

                while steps_remaining > 0
                    steps_to_run = convert(Int, min(steps_remaining, chunk_size))
                    q0, hist, chunk_number, _, _, ΣI, _, _ = run_chunk(integrator, q0, Vprime, D, Dprime, tau, stepsize, steps_to_run, hist, bin_boundaries, save_dir, repeat, chunk_number, save_traj, false, true, nothing, nothing, ΣI, nothing, x_of_y)
                    steps_remaining -= steps_to_run
                end

                # Save the q0 final values, transformed back to the original space
                q0_finals[snapshot_number, repeat] = x_of_y(q0)
            end
        end

        repeats_remaining -= repeats_to_run

        # Construct new histograms
        new_histograms = [Hist1D(q0_finals[i, :], bin_boundaries) for i in 1:number_of_snapshots]

        # Add new histograms to existing histograms
        for i in eachindex(histograms)
            histograms[i] += new_histograms[i]
        end

    end

    # Save the histograms
    save("$(save_dir)/histograms/$(string(nameof(integrator)))/space_transformed/histograms.jld2", "data", histograms)

    return histograms
end

function run_1D_finite_time_convergence_experiment(integrators, reference_integrator, reference_stepsize, num_repeats, V, D, ΔT, T, tau, stepsizes, bin_boundaries, save_dir; chunk_size=1000, save_traj=false, mu0=nothing, sigma0=nothing, time_transform=false, space_transform=false, x_of_y=nothing)
    # ΔT is the time between saving distribution snapshots
    create_experiment_folders(save_dir, integrators, reference_integrator, reference_stepsize, time_transform, space_transform, stepsizes, num_repeats, V, D, tau, bin_boundaries, chunk_size, ΔT, T)
    number_of_snapshots = Int(floor(T / ΔT))
    time_snapshots = range(0, T, length=number_of_snapshots+1)[2:end]

    # First run a high accuracy simulation with a small stepsize using the reference integrator
    reference_histograms = run_1D_finite_time_experiment(reference_integrator, num_repeats, V, D, ΔT, T, tau, reference_stepsize, bin_boundaries, save_dir; chunk_size=chunk_size, save_traj=save_traj, mu0=mu0, sigma0=sigma0)

    # Initialise error arrays
    error = zeros(length(integrators), length(stepsizes), length(time_snapshots))
    error_TT = zeros(length(integrators), length(stepsizes), length(time_snapshots))
    error_ST = zeros(length(integrators), length(stepsizes), length(time_snapshots))    

    # For each integrator, run the experiment
    for (integrator_idx, integrator) in enumerate(integrators)
        for (stepsize_idx, stepsize) in enumerate(stepsizes)
            @info "Running $(string(nameof(integrator))) experiment with $num_repeats repeats"
            histograms = run_1D_finite_time_experiment(integrator, num_repeats, V, D, ΔT, T, tau, stepsize, bin_boundaries, save_dir; chunk_size=chunk_size, save_traj=save_traj, mu0=mu0, sigma0=sigma0)

            if time_transform
                @info "Running time transformed $(string(nameof(integrator))) experiment with $num_repeats repeats"
                histograms_TT = run_1D_finite_time_experiment_time_transform(integrator, num_repeats, V, D, ΔT, T, tau, stepsize, bin_boundaries, save_dir; chunk_size=chunk_size, save_traj=save_traj, mu0=mu0, sigma0=sigma0)
            end

            if space_transform
                @info "Running space transformed $(string(nameof(integrator))) experiment with $num_repeats repeats"
                histograms_ST = run_1D_finite_time_experiment_space_transform(integrator, num_repeats, V, D, ΔT, T, tau, stepsize, bin_boundaries, save_dir; chunk_size=chunk_size, save_traj=save_traj, mu0=mu0, sigma0=sigma0, x_of_y=x_of_y)
            end

            # Compute error as a function of timesnapshot for each integrator
            estimated_errors = compute_histogram_errors(histograms, reference_histograms)
            error[integrator_idx, stepsize_idx, :] = estimated_errors

            if time_transform
                estimated_error_TT = compute_histogram_errors(histograms_TT, reference_histograms)
                error_TT[integrator_idx, stepsize_idx, :] = estimated_error_TT
            end

            if space_transform
                estimated_error_ST = compute_histogram_errors(histograms_ST, reference_histograms)
                error_ST[integrator_idx, stepsize_idx, :] = estimated_error_ST
            end
        end
    end

    # Save the error data to file
    save("$(save_dir)/results/error.jld2", "data", error)
    if time_transform
        save("$(save_dir)/results/error_TT.jld2", "data", error_TT)
    end
    if space_transform
        save("$(save_dir)/results/error_ST.jld2", "data", error_ST)
    end

    # Plot the error data
    plot_finite_time_errors(error, integrators, stepsizes, time_snapshots, save_dir, "untransformed")
    if time_transform
        plot_finite_time_errors(error_TT, integrators, stepsizes, time_snapshots, save_dir, "time_transformed")
    end
    if space_transform
        plot_finite_time_errors(error_ST, integrators, stepsizes, time_snapshots, save_dir, "space_transformed")
    end

end

end # FiniteTimeExperiments