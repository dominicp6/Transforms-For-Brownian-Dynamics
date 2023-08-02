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
import .ProbabilityUtils: compute_2D_mean_L1_error, compute_2D_invariant_distribution
import .PlottingUtils: save_and_plot, plot_histograms
import .MiscUtils: init_q0, assert_isotropic_diagonal_diffusion, is_identity_diffusion, create_directory_if_not_exists
import .DiffusionTensors: Dconst2D
import .TransformUtils: increment_g_counts2D
import .Integrators: euler_maruyama2D_identityD, naive_leimkuhler_matthews2D_identityD, stochastic_heun2D_identityD

"""
Creates necessary directories and save experiment parameters for the 2D experiment.
"""
function make_experiment2D_folders(save_dir, integrator, stepsizes, checkpoint, num_repeats, V, D, tau, x_bins, y_bins, chunk_size; T=nothing, target_uncertainty=nothing, time_transform=false)
    # Make master directory
    create_directory_if_not_exists(save_dir)

    create_directory_if_not_exists("$(save_dir)/heatmaps/$(string(nameof(integrator)))")
    for dt in stepsizes
        create_directory_if_not_exists("$(save_dir)/heatmaps/$(string(nameof(integrator)))/h=$dt")
    end

    if checkpoint
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
                        "y_bins" => y_bins,
                        "chunk_size" => chunk_size,
                        "time_transform" => time_transform)
        open("$(save_dir)/info.json", "w") do f
            JSON.print(f, metadata, 4)
        end
    end
end

"""
The `run_chunk` function runs a chunk of the 2D finite-time simulation using the specified integrator and parameters.
It performs the simulation for `steps_to_run` time steps and updates the histogram with the trajectory data.

Note: The function is typically called within the context of the main simulation loop, and its results are used for further analysis.
"""
function run_chunk2D(integrator, q0, Vprime, D, div_DDT, tau::Number, dt::Number, steps_to_run::Integer, hist, x_bins, y_bins, save_dir, repeat::Integer, chunk_number::Integer, time_transform::Bool, ΣgI, Σg, D_diag, R; identity_D=false)
    
    # Run a chunk of the simulation
    if identity_D
        # If the diffusion is identity, we can use a faster integrator
        if string(nameof(integrator)) == "euler_maruyama2D"
            q_chunk, _ = euler_maruyama2D_identityD(q0, Vprime, D, div_DDT, tau, steps_to_run, dt)
            
        elseif string(nameof(integrator)) == "naive_leimkuhler_matthews2D"
            q_chunk, _ = naive_leimkuhler_matthews2D_identityD(q0, Vprime, D, div_DDT, tau, steps_to_run, dt)

        elseif string(nameof(integrator)) == "stochastic_heun2D"
            q_chunk, _ = stochastic_heun2D_identityD(q0, Vprime, D, div_DDT, tau, steps_to_run, dt)
        else
            # Integrator not recognised
            error("Integrator $(string(nameof(integrator))) does not have a fast version for identity diffusion")
        end
    else
        q_chunk, _ = integrator(q0, Vprime, D, div_DDT, tau, steps_to_run, dt)
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

    return q0, hist, chunk_number, ΣgI, Σg
end

"""
Run a 2D finite-time experiment using the specified integrator and parameters.

# Arguments
- `integrator`: The integrator function to use for the simulation.
- `num_repeats`: Number of repeats for the experiment.
- `V`: The potential function that describes the energy landscape.
- `D`: The diffusion coefficient function that defines the noise in the system.
- `R`: The constant-coefficient matrix that is needed for the time-transformation (see paper for details).
- `T`: Total simulation time.
- `tau`: The noise strength parameter.
- `stepsizes`: An array of step sizes to be used in the simulation.
- `probabilities`: The target probabilities to compute the convergence error.
- `x_bins`: Bin boundaries for the x-axis.
- `y_bins`: Bin boundaries for the y-axis.
- `save_dir`: The directory path to save experiment results.
- `chunk_size`: Number of steps to run in each computational chunk to avoid memory issues.
- `checkpoint`: If true, save intermediate results in checkpoints.
- `q0`: The initial position of the trajectory. If not provided, it will be randomly initialized.
- `time_transform`: If true, apply time transformation to the potential and diffusion.

# Returns
- `convergence_errors`: A matrix containing convergence errors for each step size and repeat.

# Details
This function runs a 1D finite-time experiment with the specified integrator and system parameters. It supports various configurations, including time and space transformations. 
The experiment is repeated `num_repeats` times, each time with different initial conditions. For each combination of step size and repeat, the weak error w.r.t. the invariant distribution is computed.

Note: The `V` and `D` functions may be modified internally to implement time or space transformations, based on the provided `time_transform` and `space_transform` arguments.
"""
function run_2D_experiment(integrator, num_repeats, V, D, T, R, tau, stepsizes, probabilities, x_bins, y_bins, save_dir; chunk_size=10000000, checkpoint=false, q0=nothing, time_transform=false)
    
    make_experiment2D_folders(save_dir, integrator, stepsizes, checkpoint, num_repeats, V, D, tau, x_bins, y_bins, chunk_size, T=T, target_uncertainty=nothing, time_transform=time_transform)
   
    # [For time-transformed integrators] Modify the potential and diffusion functions appropriately (see paper for details)
    D_diag = nothing
    transform_potential_and_diffusion!(V, D, D_diag, R, tau, time_transform)

    # Verify whether the diffusion is identity
    identity_diffusion = is_identity_diffusion(D)

    # Compute symbolic derivatives of the potential and diffusion
    Vprime = differentiate2D(V)
    DDT = (x,y) -> D(x,y) * Base.transpose(D(x,y))
    div_DDT = symbolic_matrix_divergence2D(DDT)

    # Initialise empty data arrays
    convergence_errors = zeros(length(stepsizes), num_repeats)
    histogram_data = Matrix{Hist2D}(undef, length(stepsizes), num_repeats)

    Threads.@threads for repeat in ProgressBar(1:num_repeats)
        # set the random seed for reproducibility
        Random.seed!(repeat) 

        # If no initial position is provided, randomly initialise
        q0 = init_q0(q0, dim=2)

        # Run the simulation for each specified step size
        for (stepsize_idx, dt) in enumerate(stepsizes)

            steps_remaining = floor(T / dt)                 
            total_samples = Int(steps_remaining)    
            chunk_number = 0                                   

            # Create a zeros array of the correct size for the histogram
            num_x_bins = length(x_bins) - 1
            num_y_bins = length(y_bins) - 1
            zeros_array = zeros(Int64, num_x_bins, num_y_bins)

            # For time-transformed integrators, we need to keep track of the following quantities for reweighting
            ΣgI = zeros(Int64, num_x_bins, num_y_bins)     
            Σg = 0.0  

            hist = Hist2D(zeros_array, (x_bins, y_bins))                 # histogram of the trajectory

            while steps_remaining > 0
                # Run steps in chunks to minimise memory footprint
                steps_to_run = convert(Int, min(steps_remaining, chunk_size))
                q0, hist, chunk_number, ΣgI, Σg = run_chunk2D(integrator, q0, Vprime, D, div_DDT, tau, dt, steps_to_run, hist, x_bins, y_bins, save_dir, repeat, chunk_number, time_transform, ΣgI, Σg, D_diag, R, identity_D=identity_diffusion)
                steps_remaining -= steps_to_run
            end

            # Compute the convergence error
            if time_transform
                empirical_probabilities = ΣgI ./ Σg
                convergence_errors[stepsize_idx, repeat] = compute_2D_mean_L1_error(empirical_probabilities, probabilities)
            else
                convergence_errors[stepsize_idx, repeat] = compute_2D_mean_L1_error(hist, probabilities, total_samples)
            end

            histogram_data[stepsize_idx, repeat] = hist

            if checkpoint
                # Save the histogram
                save("$(save_dir)/checkpoints/$(string(nameof(integrator)))/h=$dt/$(repeat).jld2", "data", hist)
            end
        end
    end

    # Save the error data and plot
    save_and_plot(integrator, convergence_errors, stepsizes, save_dir)
    plot_histograms(integrator, histogram_data, stepsizes, save_dir)

    # Print the mean and standard deviation of the L1 errors
    @info "Mean L1 errors: $(mean(convergence_errors, dims=2))"
    @info "Standard deviation of L1 errors: $(std(convergence_errors, dims=2))"

    return convergence_errors
end


function transform_potential_and_diffusion!(V, D, D_diag, R, tau, time_transform)
    if time_transform 
        assert_isotropic_diagonal_diffusion(D)
        D_diag = (x,y) -> D(x,y)[1,1]
        # Transform the potential so that the diffusion is constant
        V = (x,y) -> V(R[1,1]*x + R[1,2]*y, R[2,1]*x + R[2,2]*y) - 2 * tau * log(D_diag(R[1,1]*x + R[1,2]*y, R[2,1]*x + R[2,2]*y))
        D = Dconst2D
    end
end

"""
Run a master 2D experiment with multiple integrators.

Parameters:
- `integrators`: An array of integrators to be used in the experiments.
- `num_repeats`: Number of experiment repeats to perform for each integrator.
- `V`: Potential function V(x) representing the energy landscape.
- `D`: Diffusion function D(x) representing the diffusion coefficient.
- `R`: The constant-coefficient matrix that is needed for the time-transformation (see paper for details).
- `T`: Total time for the simulation.
- `tau`: The noise strength parameter.
- `stepsizes`: An array of time step sizes to be used in the simulation.
- `xmin`: The minimum x-coordinate of the domain.
- `ymin`: The minimum y-coordinate of the domain.
- `xmax`: The maximum x-coordinate of the domain.
- `ymax`: The maximum y-coordinate of the domain.
- `n_bins`: The number of bins to be used in the histogram (in each dimension).
- `save_dir`: The directory where results and time convergence data will be saved.
- `chunk_size`: Number of simulation steps to be run in each chunk. Default is 10000000.
- `checkpoint`: A boolean flag indicating whether to save checkpoints. Default is false.
- `q0`: The initial configuration for the simulation. Default is nothing, which generates a random configuration.
- `time_transform`: A boolean flag indicating whether to apply time transformation. Default is false.

Returns:
- The function saves the results of each experiment in the specified `save_dir` and also saves the time convergence data in a file named "time.json".
"""
function master_2D_experiment(integrators, num_repeats, V, D, T, R, tau, stepsizes, xmin, ymin, xmax, ymax, n_bins, save_dir; chunk_size=10000000, checkpoint=false, q0=nothing, time_transform=false)
    to = TimerOutput()

    @info "Computing Expected Probabilities"
    probabilities, x_bins, y_bins, n_bins = compute_2D_invariant_distribution(V, tau, xmin, ymin, xmax, ymax, n_bins)
    
    @info "Running Experiments"
    for integrator in integrators
        @info "Running $(string(nameof(integrator))) experiment"
        @timeit to "Exp$(string(nameof(integrator)))" begin 
            _ = run_2D_experiment(integrator, num_repeats, V, D, T, R, tau, stepsizes, probabilities, x_bins, y_bins, save_dir, chunk_size=chunk_size, checkpoint=checkpoint, q0=q0, time_transform=time_transform)
        end
    end

    # save the time convergence_data
    open("$(save_dir)/time.json", "w") do io
        JSON.print(io, TimerOutputs.todict(to), 4)
    end
end

end # module