module Experiments
include("calculus.jl")
include("potentials.jl")
include("diffusionTensors.jl")
include("utils.jl")
include("transform_utils.jl")
using HCubature, QuadGK, FHist, JLD2, Statistics, .Threads, ProgressBars, Plots, JSON, Random, StatsBase, HDF5, TimerOutputs
import .Calculus: differentiate1D, differentiateND, differentiate2D, symbolic_matrix_divergence2D
import .DiffusionTensors: Dconst1D
import .Utils: compute_1D_mean_L1_error, compute_2D_mean_L1_error, compute_1D_probabilities, compute_2D_probabilities
import .TransformUtils: increment_g_counts, increment_I_counts
export run_1D_experiment, run_2D_experiment, master_1D_experiment, run_1D_experiment_until_given_uncertainty, master_2D_experiment

function make_experiment_folders(save_dir, integrator, stepsizes, checkpoint, save_traj, num_repeats, V, D, tau, x_bins, chunk_size, time_transform, space_transform; T=nothing, target_uncertainty=nothing)
    # Make experiment folders
    if !isdir(save_dir)
        mkdir(save_dir)
        @info "Created directory $(save_dir)"
    end

    if checkpoint && !isdir("$(save_dir)/checkpoints")
        mkdir("$(save_dir)/checkpoints")
    end

    if checkpoint && !isdir("$(save_dir)/checkpoints/$(string(nameof(integrator)))")
        mkdir("$(save_dir)/checkpoints/$(string(nameof(integrator)))")
        for dt in stepsizes
            mkdir("$(save_dir)/checkpoints/$(string(nameof(integrator)))/h=$dt")
        end
    end

    if save_traj && !isdir("$(save_dir)/trajectories")
        mkdir("$(save_dir)/trajectories")
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

function save_and_plot(integrator, convergence_data, stepsizes, save_dir; xlabel="dt", ylabel="Mean L1 error", error_in_mean=false)
    @info "Saving data"
    h5write("$(save_dir)/$(integrator).h5", "data", convergence_data)

    number_of_repeats = size(convergence_data, 2)

    # Plot (dim 1 is the step size, dim 2 is the repeat)
    plot(stepsizes, mean(convergence_data, dims=2), title=string(nameof(integrator)), xlabel=xlabel, ylabel=ylabel, xscale=:log10, yscale=:log10, label="")
    if error_in_mean
        plot!(stepsizes, mean(convergence_data, dims=2)+std(convergence_data, dims=2)/sqrt(number_of_repeats), ls=:dash, lc=:black, label="")
        plot!(stepsizes, mean(convergence_data, dims=2)-std(convergence_data, dims=2)/sqrt(number_of_repeats), ls=:dash, lc=:black, label="")
    else
        plot!(stepsizes, mean(convergence_data, dims=2)+std(convergence_data, dims=2), ls=:dash, lc=:black, label="")
        plot!(stepsizes, mean(convergence_data, dims=2)-std(convergence_data, dims=2), ls=:dash, lc=:black, label="")
    end
    savefig("$(save_dir)/$(integrator).png")
end

function run_chunk(integrator, q0, Vprime, D, Dprime, tau::Number, dt::Number, steps_to_run::Integer, hist, bin_boundaries, save_dir, repeat::Integer, chunk_number::Integer, save_traj::Bool, time_transform::Bool, space_transform:: Bool, ΣgI::Vector, Σg::Float64, ΣI::Vector, original_D, x_of_y)
    # Run a chunk of the simulation
    q_chunk = integrator(q0, Vprime, D, Dprime, tau, steps_to_run, dt)
    
    # [For time-transformed integrators] Increment g counts
    if time_transform
        ΣgI, Σg =  increment_g_counts(q_chunk, original_D, bin_boundaries, ΣgI, Σg)
    end

    # [For space-transformed integrators] Increment I counts
    if space_transform
        ΣI = increment_I_counts(q_chunk, x_of_y, bin_boundaries, ΣI)
    end

    # Set the initial condition to the last value of the previous chunk
    q0 = copy(q_chunk[end])  

    # Update the number of steps left to run
    hist += Hist1D(q_chunk, bin_boundaries)
    chunk_number += 1

    # [Optional] Save the chunk
    if save_traj
        h5write("$(save_dir)/trajectories/$(string(nameof(integrator)))/h=$dt/$(repeat).$(chunk_number).h5", "data", q_chunk)
    end

    return q0, hist, chunk_number, ΣgI, Σg, ΣI
end

function init_q0(q0)
    if q0 === nothing
        q0 = randn()
    end
    return q0
end

function run_1D_experiment(integrator, num_repeats, V, D, T, tau, stepsizes, probabilities, bin_boundaries, save_dir; chunk_size=10000000, checkpoint=false, q0=nothing, save_traj=false, time_transform=false, space_transform=false, x_of_y=nothing)
    make_experiment_folders(save_dir, integrator, stepsizes, checkpoint, save_traj, num_repeats, V, D, tau, bin_boundaries, chunk_size, time_transform, space_transform, T=T)
    
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
    convergence_data = zeros(length(stepsizes), num_repeats)

    for repeat in ProgressBar(1:num_repeats)

        Random.seed!(repeat) # set the random seed for reproducibility
        q0 = init_q0(q0)

        for (stepsize_idx, dt) in enumerate(stepsizes)

            # Initialise for this step size
            steps_remaining = floor(T / dt)                   # total number of steps
            chunk_number = 0                                  # number of chunks run so far
            hist = Hist1D([], bin_boundaries)                 # histogram of the trajectory
            
            # For time-transformed integrators
            ΣgI = zeros(length(bin_boundaries)-1)            
            Σg = 0.0   

            # For space-transformed integrators
            ΣI = zeros(length(bin_boundaries)-1)

            # Run steps in chunks to avoid memory issues
            while steps_remaining > 0
                steps_to_run = convert(Int, min(steps_remaining, chunk_size))
                q0, hist, chunk_number, ΣgI, Σg, ΣI = run_chunk(integrator, q0, Vprime, D, Dprime, tau, dt, steps_to_run, hist, bin_boundaries, save_dir, repeat, chunk_number, save_traj, time_transform, space_transform, ΣgI, Σg, ΣI, original_D, x_of_y)
                steps_remaining -= steps_to_run
            end

            if time_transform
                empirical_probabilities = ΣgI ./ Σg
                convergence_data[stepsize_idx, repeat] = compute_1D_mean_L1_error(empirical_probabilities, probabilities)
            elseif space_transform
                empirical_probabilities = ΣI ./ floor(T/dt)
                convergence_data[stepsize_idx, repeat] = compute_1D_mean_L1_error(empirical_probabilities, probabilities)
            else
                convergence_data[stepsize_idx, repeat] = compute_1D_mean_L1_error(hist, probabilities)
            end

            if checkpoint
                save("$(save_dir)/checkpoints/$(string(nameof(integrator)))/h=$dt/$(repeat).jld2", "data", hist)
            end
        end
    end

    save_and_plot(integrator, convergence_data, stepsizes, save_dir)

    # Print the mean and standard deviation of the L1 errors
    @info "Mean L1 errors: $(mean(convergence_data, dims=2))"
    @info "Standard deviation of L1 errors: $(std(convergence_data, dims=2))"

    return convergence_data
end

function run_1D_experiment_until_given_uncertainty(integrator, num_repeats, V, D, tau, stepsizes, probabilities, bin_boundaries, save_dir, target_uncertainty; chunk_size=10000, checkpoint=false, q0=nothing, save_traj=false, time_transform=false, space_transform=false, x_of_y=nothing)
    make_experiment_folders(save_dir, integrator, stepsizes, checkpoint, save_traj, num_repeats, V, D, tau, bin_boundaries, chunk_size, time_transform, space_transform, target_uncertainty=target_uncertainty)
    
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
                    error = compute_1D_mean_L1_error(hist, probabilities)
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

# function run_2D_experiment(integrator, num_repeats, V, D, T, tau, stepsizes, probabilities, x_bins, y_bins, n_bins, checkpoint=true)
#     convergence_data = zeros(length(stepsizes), num_repeats)
#     for repeat in 1:num_repeats
#         q0 = randn(2)
#         for (step_idx, dt) in enumerate(stepsizes)
#             steps_to_run = floor(Int, T / dt)
#             q_traj = integrator(q0, V, D, tau, steps_to_run, dt)
#             has_nan = any(isnan, q_traj)

#             if has_nan
#                 error("NaN value encountered in q_traj. Step size: $dt")
#             end
#             h2, mean_error = compute_2D_mean_L1_error(q_traj, probabilities, x_bins, y_bins, n_bins)
#             convergence_data[step_idx, repeat] = mean_error
#             if checkpoint
#                 save("q_traj_h=$(dt)_r$(repeat).jld2", "data", h2)
#             end
#         end
#     end
#     return convergence_data
# end

function make_experiment2D_folders(save_dir, integrator, stepsizes, checkpoint, save_traj, num_repeats, V, D, tau, x_bins, y_bins, chunk_size; T=nothing, target_uncertainty=nothing)
    # Make experiment folders
    if !isdir(save_dir)
        mkdir(save_dir)
        @info "Created directory $(save_dir)"
    end

    if checkpoint && !isdir("$(save_dir)/checkpoints")
        mkdir("$(save_dir)/checkpoints")
    end

    if checkpoint && !isdir("$(save_dir)/checkpoints/$(string(nameof(integrator)))")
        mkdir("$(save_dir)/checkpoints/$(string(nameof(integrator)))")
        for dt in stepsizes
            mkdir("$(save_dir)/checkpoints/$(string(nameof(integrator)))/h=$dt")
        end
    end

    if save_traj && !isdir("$(save_dir)/trajectories")
        mkdir("$(save_dir)/trajectories")
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
                        "y_bins" => y_bins,
                        "chunk_size" => chunk_size)
        open("$(save_dir)/info.json", "w") do f
            JSON.print(f, metadata, 4)
        end
    end
end


function run_chunk2D(integrator, q0, Vprime, DDT, div_DDT, tau::Number, dt::Number, steps_to_run::Integer, hist, x_bins, y_bins, save_dir, repeat::Integer, chunk_number::Integer, save_traj::Bool)
    # Run a chunk of the simulation
    q_chunk = integrator(q0, Vprime, DDT, div_DDT, tau, steps_to_run, dt)

    # Set the initial condition to the last value of the previous chunk
    q0 = copy(q_chunk[end])  

    # Update the number of steps left to run
    hist += Hist2D(q_chunk, (x_bins, y_bins))
    chunk_number += 1

    # [Optional] Save the chunk
    if save_traj
        h5write("$(save_dir)/trajectories/$(string(nameof(integrator)))/h=$dt/$(repeat).$(chunk_number).h5", "data", q_chunk)
    end

    return q0, hist, chunk_number
end

function run_2D_experiment(integrator, num_repeats, V, D, T, tau, stepsizes, probabilities, x_bins, y_bins, save_dir; chunk_size=10000000, checkpoint=false, q0=nothing, save_traj=false)
    make_experiment2D_folders(save_dir, integrator, stepsizes, checkpoint, save_traj, num_repeats, V, D, tau, x_bins, y_bins, chunk_size, T=T, target_uncertainty=nothing)
   
    Vprime = differentiate2D(V)
    DDT = (x,y) -> D(x,y) * Base.transpose(D(x,y))
    div_DDT = symbolic_matrix_divergence2D(DDT)

    

    @info "Running $(string(nameof(integrator))) experiment with $num_repeats repeats"
    convergence_data = zeros(length(stepsizes), num_repeats)

    for repeat in ProgressBar(1:num_repeats)

        Random.seed!(repeat) # set the random seed for reproducibility
        q0 = init_q0(q0)

        for (stepsize_idx, dt) in enumerate(stepsizes)

            # Initialise for this step size
            steps_remaining = floor(T / dt)                   # total number of steps
            chunk_number = 0                                  # number of chunks run so far
            hist = Hist2D([], (x_bins, y_bins))                 # histogram of the trajectory

            # Run steps in chunks to avoid memory issues
            while steps_remaining > 0
                steps_to_run = convert(Int, min(steps_remaining, chunk_size))
                q0, hist, chunk_number = run_chunk2D(integrator, q0, Vprime, DDT, div_DDT, tau, dt, steps_to_run, hist, x_bins, y_bins, save_dir, repeat, chunk_number, save_traj)
                steps_remaining -= steps_to_run
            end

            convergence_data[stepsize_idx, repeat] = compute_2D_mean_L1_error(hist, probabilities)

            if checkpoint
                save("$(save_dir)/checkpoints/$(string(nameof(integrator)))/h=$dt/$(repeat).jld2", "data", hist)
            end
        end
    end

    save_and_plot(integrator, convergence_data, stepsizes, save_dir)

    # Print the mean and standard deviation of the L1 errors
    @info "Mean L1 errors: $(mean(convergence_data, dims=2))"
    @info "Standard deviation of L1 errors: $(std(convergence_data, dims=2))"

    return convergence_data
end


function master_2D_experiment(integrators, num_repeats, V, D, T, tau, stepsizes, xmin, ymin, xmax, ymax, n_bins, save_dir; chunk_size=10000000, checkpoint=false, q0=nothing, save_traj=false)
    to = TimerOutput()

    @info "Computing Expected Probabilities"
    probabilities, x_bins, y_bins, n_bins = compute_2D_probabilities(V, tau, xmin, ymin, xmax, ymax, n_bins)
    
    @info "Running Experiments"
    for integrator in integrators
        @info "Running $(string(nameof(integrator))) experiment"
        @timeit to "Exp$(string(nameof(integrator)))" begin 
            convergence_data = run_2D_experiment(integrator, num_repeats, V, D, T, tau, stepsizes, probabilities, x_bins, y_bins, save_dir, chunk_size=chunk_size, checkpoint=checkpoint, q0=q0, save_traj=save_traj)
        end
    end

    # save the time convergence_data
    open("$(save_dir)/time.json", "w") do io
        JSON.print(io, TimerOutputs.todict(to), 4)
    end
end

end # module