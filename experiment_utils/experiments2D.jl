module Experiments2D
include("../general_utils/calculus.jl")
include("../general_utils/potentials.jl")
include("../general_utils/probability_utils.jl")
include("../general_utils/plotting_utils.jl")
include("../general_utils/misc_utils.jl")
include("../general_utils/transform_utils.jl")
include("../general_utils/diffusion_tensors.jl")
include("../general_utils/integrators.jl")
using HCubature, QuadGK, FHist, JLD2, Statistics, .Threads, ProgressBars, JSON, Random, StatsBase, TimerOutputs, LinearAlgebra
import .Calculus: differentiate2D, symbolic_matrix_divergence2D
import .ProbabilityUtils: compute_2D_mean_L1_error, compute_2D_probabilities
import .PlottingUtils: save_and_plot, plot_histograms
import .MiscUtils: init_q0, assert_isotropic_diagonal_diffusion
import .DiffusionTensors: Dconst2D
import .TransformUtils: increment_g_counts2D
import .Integrators: euler_maruyama2D_identityD, naive_leimkuhler_matthews2D_identityD, stochastic_heun2D_identityD


function make_experiment2D_folders(save_dir, integrator, stepsizes, checkpoint, save_traj, num_repeats, V, D, tau, x_bins, y_bins, chunk_size; T=nothing, target_uncertainty=nothing, time_transform=false)
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
                        "chunk_size" => chunk_size,
                        "time_transform" => time_transform)
        open("$(save_dir)/info.json", "w") do f
            JSON.print(f, metadata, 4)
        end
    end
end


function run_chunk2D(integrator, q0, Vprime, D, div_DDT, tau::Number, dt::Number, steps_to_run::Integer, hist, x_bins, y_bins, save_dir, repeat::Integer, chunk_number::Integer, save_traj::Bool, time_transform::Bool, ΣgI, Σg, D_diag, R; identity_D=false)
    
    # Run a chunk of the simulation
    if identity_D
        # If the diffusion is identity, we can use a faster integrator
        if string(nameof(integrator)) == "euler_maruyama2D"
            q_chunk = euler_maruyama2D_identityD(q0, Vprime, D, div_DDT, tau, steps_to_run, dt)
            
        elseif string(nameof(integrator)) == "naive_leimkuhler_matthews2D"
            q_chunk = naive_leimkuhler_matthews2D_identityD(q0, Vprime, D, div_DDT, tau, steps_to_run, dt)

        elseif string(nameof(integrator)) == "stochastic_heun2D"
            q_chunk = stochastic_heun2D_identityD(q0, Vprime, D, div_DDT, tau, steps_to_run, dt)
        else
            # Integrator not recognised
            error("Integrator $(string(nameof(integrator))) does not have a fast version for identity diffusion")
        end
    else
        q_chunk = integrator(q0, Vprime, D, div_DDT, tau, steps_to_run, dt)
    end

    # [For time-transformed integrators] Increment g counts
    if time_transform
        ΣgI, Σg =  increment_g_counts2D(q_chunk, D_diag, x_bins, y_bins, ΣgI, Σg, R)
    end

    # Set the initial condition to the last value of the previous chunk
    q0 = copy(q_chunk[:, end])  

    # Update the number of steps left to run
    hist += Hist2D((q_chunk[1,:], q_chunk[2,:]), (x_bins, y_bins))
    chunk_number += 1

    # [Optional] Save the chunk
    if save_traj
        h5write("$(save_dir)/trajectories/$(string(nameof(integrator)))/h=$dt/$(repeat).$(chunk_number).h5", "data", q_chunk)
    end

    return q0, hist, chunk_number, ΣgI, Σg
end

function run_2D_experiment(integrator, num_repeats, V, D, T, R, tau, stepsizes, probabilities, x_bins, y_bins, save_dir; chunk_size=10000000, checkpoint=false, q0=nothing, save_traj=false, time_transform=false)
    make_experiment2D_folders(save_dir, integrator, stepsizes, checkpoint, save_traj, num_repeats, V, D, tau, x_bins, y_bins, chunk_size, T=T, target_uncertainty=nothing, time_transform=time_transform)
   
    original_V = V
    original_D = D
    D_diag = nothing

    if time_transform 
        assert_isotropic_diagonal_diffusion(D)
        D_diag = (x,y) -> D(x,y)[1,1]
        # Transform the potential so that the diffusion is constant
        V = (x,y) -> original_V(R[1,1]*x + R[1,2]*y, R[2,1]*x + R[2,2]*y) - 2 * tau * log(D_diag(R[1,1]*x + R[1,2]*y, R[2,1]*x + R[2,2]*y))
        D = Dconst2D
    end

    if D(0,0) == D(1,1) == D(-0.2354345, 0.21267) == I
        identity_D = true
    else
        identity_D = false
    end

    println(identity_D)

    Vprime = differentiate2D(V)
    DDT = (x,y) -> D(x,y) * Base.transpose(D(x,y))
    div_DDT = symbolic_matrix_divergence2D(DDT)

    @info "Running $(string(nameof(integrator))) experiment with $num_repeats repeats"
    convergence_data = zeros(length(stepsizes), num_repeats)
    convergence_data2 = zeros(length(stepsizes), num_repeats)
    histogram_data = Matrix{Hist2D}(undef, length(stepsizes), num_repeats)

    Threads.@threads for repeat in ProgressBar(1:num_repeats)

        Random.seed!(repeat) # set the random seed for reproducibility
        q0 = init_q0(q0, dim=2)

        for (stepsize_idx, dt) in enumerate(stepsizes)

            # Initialise for this step size
            steps_remaining = floor(T / dt)                   # total number of steps
            total_samples = Int(steps_remaining)              # total number of steps
            chunk_number = 0                                  # number of chunks run so far             

            # Create an array of zeros with the desired shape
            num_x_bins = length(x_bins) - 1
            num_y_bins = length(y_bins) - 1
            zeros_array = zeros(Int64, num_x_bins, num_y_bins)

            # For time-transformed integrators
            ΣgI = zeros(Int64, num_x_bins, num_y_bins)     
            Σg = 0.0  

            hist = Hist2D(zeros_array, (x_bins, y_bins))                 # histogram of the trajectory

            # Run steps in chunks to avoid memory issues
            while steps_remaining > 0
                steps_to_run = convert(Int, min(steps_remaining, chunk_size))
                q0, hist, chunk_number, ΣgI, Σg = run_chunk2D(integrator, q0, Vprime, D, div_DDT, tau, dt, steps_to_run, hist, x_bins, y_bins, save_dir, repeat, chunk_number, save_traj, time_transform, ΣgI, Σg, D_diag, R, identity_D=identity_D)
                steps_remaining -= steps_to_run
            end

            if time_transform
                empirical_probabilities = ΣgI ./ Σg
                convergence_data[stepsize_idx, repeat] = compute_2D_mean_L1_error(empirical_probabilities, probabilities)
            else
                convergence_data[stepsize_idx, repeat] = compute_2D_mean_L1_error(hist, probabilities, total_samples)
            end

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


function master_2D_experiment(integrators, num_repeats, V, D, T, R, tau, stepsizes, xmin, ymin, xmax, ymax, n_bins, save_dir; chunk_size=10000000, checkpoint=false, q0=nothing, save_traj=false, time_transform=false)
    to = TimerOutput()

    @info "Computing Expected Probabilities"
    probabilities, x_bins, y_bins, n_bins = compute_2D_probabilities(V, tau, xmin, ymin, xmax, ymax, n_bins)
    
    @info "Running Experiments"
    for integrator in integrators
        @info "Running $(string(nameof(integrator))) experiment"
        @timeit to "Exp$(string(nameof(integrator)))" begin 
            convergence_data = run_2D_experiment(integrator, num_repeats, V, D, T, R, tau, stepsizes, probabilities, x_bins, y_bins, save_dir, chunk_size=chunk_size, checkpoint=checkpoint, q0=q0, save_traj=save_traj, time_transform=time_transform)
        end
    end

    # save the time convergence_data
    open("$(save_dir)/time.json", "w") do io
        JSON.print(io, TimerOutputs.todict(to), 4)
    end
end

end # module