module Utils
using HCubature, QuadGK, FHist, JLD2, .Threads, ProgressBars
export compute_1D_probabilities, compute_2D_probabilities, compute_1D_mean_L1_error, compute_2D_mean_L1_error, run_1D_experiment, run_2D_experiment

function compute_2D_probabilities(V, tau, xmin, ymin, xmax, ymax, n_bins)
    # Compute the expected counts in a 2D histogram of the configuration space

    # Histogram parameters
    x_bins = range(xmin, xmax, length=n_bins+1)
    y_bins = range(ymin, ymax, length=n_bins+1)

    f(q) = exp(-V(q) / tau) 
    prob = zeros(n_bins, n_bins)
    # Compute the integral of f over the entire configuration space (approximated by integral over a large square)
    integral_of_f, err = hcubature(f, [-5, -5], [5, 5])
    println("Integral of f: $integral_of_f")
    println("Error in integral of f: $err")
    for i = 1:n_bins, j = 1:n_bins
        x_min, x_max = x_bins[i], x_bins[i+1]
        y_min, y_max = y_bins[j], y_bins[j+1]
        # integrate pdf over the bin        
        prob[i, j], _ = hcubature(f, (x_min, y_min), (x_max, y_max))                  
    end

    prob /= integral_of_f  # scale by integral of f to get normalised probabilities

    return prob, x_bins, y_bins, n_bins
end


function compute_1D_probabilities(V, tau, xmin, ymin, n_bins)
    # Compute the expected counts in a 1D histogram of the configuration space

    # Histogram parameters
    x_bins = range(xmin, ymin, length=n_bins+1)

    f(q) = exp(-V(q) / tau) 
    prob = zeros(n_bins)
    # Compute the integral of f over the entire configuration space (approximated by integral over a large square)
    integral_of_f, err = quadgk(f, -5, 5)
    println("Integral of f: $integral_of_f")
    println("Error in integral of f: $err")
    for i = 1:n_bins
        x_min, x_max = x_bins[i], x_bins[i+1]
        # integrate pdf over the bin        
        prob[i], _ = quadgk(f, x_min, x_max)                  
    end

    prob /= integral_of_f  # scale by integral of f to get normalised probabilities

    return prob, x_bins, n_bins
end


function compute_2D_mean_L1_error(q_traj, probabilities, x_bins, y_bins, n_bins)
    # Compute the mean L1 error between the expected probability and
    # the actual probability distributions across all bins of a 2D histogram

    # Compute the histogram
    h2 = Hist2D(collect(eachrow(q_traj)), (x_bins, y_bins))

    # Compute the means L1 error between the expected probability and the actual probability distributions
    mean_error = sum(abs.(bincounts(h2)/length(q_traj[1, :]) .- probabilities)) / (n_bins * n_bins)

    return h2, mean_error
end


function compute_1D_mean_L1_error(q_traj, probabilities, x_bins, n_bins)
    # Compute the mean L1 error between the expected probability and
    # the actual probability distributions across all bins of a 1D histogram

    # Compute the histogram
    h1 = Hist1D(q_traj, x_bins)

    # Compute the means L1 error between the expected probability and the actual probability distributions
    mean_error = sum(abs.(bincounts(h1)/length(q_traj) .- probabilities)) / (n_bins)

    return h1, mean_error
end


function run_2D_experiment(integrator, num_repeats, V, D, T, tau, stepsizes, probabilities, x_bins, y_bins, n_bins, checkpoint=true)
    convergence_data = zeros(length(stepsizes), num_repeats)
    for repeat in 1:num_repeats
        q0 = randn(2)
        for (step_idx, dt) in enumerate(stepsizes)
            q_traj = integrator(q0, V, D, tau, T, dt, repeat)
            h2, mean_error = compute_2D_mean_L1_error(q_traj, probabilities, x_bins, y_bins, n_bins)
            convergence_data[step_idx, repeat] = mean_error
            if checkpoint
                save("q_traj_h=$(dt)_r$(repeat).jld2", "data", h2)
            end
        end
    end
    return convergence_data
end


function run_1D_experiment(integrator, num_repeats, V, D, T, tau, stepsizes, probabilties, x_bins, n_bins, checkpoint=true)
    convergence_data = zeros(length(stepsizes), num_repeats)
    Threads.@threads for repeat in ProgressBar(1:num_repeats)
        q0 = randn()
        for (step_idx, dt) in enumerate(stepsizes)
            q_traj = integrator(q0, V, D, tau, T, dt, repeat)
            h1, mean_error = compute_1D_mean_L1_error(q_traj, probabilties, x_bins, n_bins)
            convergence_data[step_idx, repeat] = mean_error
            if checkpoint
                save("q_traj_h=$(dt)_r$(repeat).jld2", "data", h1)
            end
        end
    end
    return convergence_data
end

end # module Utils