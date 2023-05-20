module Experiments2D
include("calculus.jl")
include("potentials.jl")
include("utils.jl")
include("transform_utils.jl")
using HCubature, QuadGK, FHist, JLD2, Statistics, .Threads, ProgressBars, JSON, Random, StatsBase, TimerOutputs
import .Calculus: differentiate2D, symbolic_matrix_divergence2D
import .Utils: compute_2D_mean_L1_error, compute_2D_probabilities, save_and_plot, init_q0, plot_histograms


function make_experiment2D_folders(save_dir, integrator, stepsizes, checkpoint, save_traj, num_repeats, V, D, tau, x_bins, y_bins, chunk_size; T=nothing, target_uncertainty=nothing)
    # Make experiment folders
    if !isdir(save_dir)
        mkdir(save_dir)
        @info "Created directory $(save_dir)"
    end

    if !isdir("$(save_dir)/heatmaps")
        mkdir("$(save_dir)/heatmaps")
    end

    if !isdir("$(save_dir)/heatmaps/$(string(nameof(integrator)))")
        mkdir("$(save_dir)/heatmaps/$(string(nameof(integrator)))")
        for dt in stepsizes
            mkdir("$(save_dir)/heatmaps/$(string(nameof(integrator)))/h=$dt")
        end
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


function run_chunk2D(integrator, q0, Vprime, D, div_DDT, tau::Number, dt::Number, steps_to_run::Integer, hist, x_bins, y_bins, save_dir, repeat::Integer, chunk_number::Integer, save_traj::Bool)
    # Run a chunk of the simulation
    q_chunk = integrator(q0, Vprime, D, div_DDT, tau, steps_to_run, dt)

    # Set the initial condition to the last value of the previous chunk
    q0 = copy(q_chunk[:, end])  

    # Update the number of steps left to run
    hist += Hist2D((q_chunk[1,:], q_chunk[2,:]), (x_bins, y_bins))
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
    histogram_data = Matrix{Hist2D}(undef, length(stepsizes), num_repeats)

    Threads.@threads for repeat in ProgressBar(1:num_repeats)

        Random.seed!(repeat) # set the random seed for reproducibility
        q0 = init_q0(q0, dim=2)

        for (stepsize_idx, dt) in enumerate(stepsizes)

            # Initialise for this step size
            steps_remaining = floor(T / dt)                   # total number of steps
            chunk_number = 0                                  # number of chunks run so far

            # Create an array of zeros with the desired shape
            num_x_bins = length(x_bins) - 1
            num_y_bins = length(y_bins) - 1
            zeros_array = zeros(Int64, num_x_bins, num_y_bins)

            hist = Hist2D(zeros_array, (x_bins, y_bins))                 # histogram of the trajectory

            # Run steps in chunks to avoid memory issues
            while steps_remaining > 0
                steps_to_run = convert(Int, min(steps_remaining, chunk_size))
                q0, hist, chunk_number = run_chunk2D(integrator, q0, Vprime, D, div_DDT, tau, dt, steps_to_run, hist, x_bins, y_bins, save_dir, repeat, chunk_number, save_traj)
                steps_remaining -= steps_to_run
            end

            convergence_data[stepsize_idx, repeat] = compute_2D_mean_L1_error(hist, probabilities)
            histogram_data[stepsize_idx, repeat] = hist

            if checkpoint
                save("$(save_dir)/checkpoints/$(string(nameof(integrator)))/h=$dt/$(repeat).jld2", "data", hist)
            end
        end
    end

    save_and_plot(integrator, convergence_data, stepsizes, save_dir)
    plot_histograms(integrator, histogram_data, stepsizes, save_dir)

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