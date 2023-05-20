module Utils
using HCubature, QuadGK, FHist, HDF5, Statistics, StatsBase, Plots
export compute_1D_probabilities, compute_2D_probabilities, compute_1D_mean_L1_error, compute_2D_mean_L1_error, save_and_plot, init_q0, plot_heatmap, plot_histograms

function compute_2D_probabilities(V, tau, xmin, ymin, xmax, ymax, n_bins)
    # Compute the expected counts in a 2D histogram of the configuration space

    # Histogram parameters
    x_bins = range(xmin, xmax, length=n_bins+1)
    y_bins = range(ymin, ymax, length=n_bins+1)

    function V1arg(q)
        x, y = q
        return V(x, y)
    end

    f(q) = exp(-V1arg(q) / tau) 
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


function compute_1D_probabilities(V, tau, bins; configuration_space=(-12,12))
    # Compute the expected counts in a 1D histogram of the configuration space

    n_bins = length(bins) - 1

    f(q) = exp(-V(q) / tau) 
    prob = zeros(length(bins) - 1)
    # Compute the integral of f over the entire configuration space (approximated by integral over a large square)
    integral_of_f, err = quadgk(f, configuration_space[1], configuration_space[2])
    println("Integral of f: $integral_of_f")
    println("Error in integral of f: $err")
    for i = 1:n_bins
        x_min, x_max = bins[i], bins[i+1]
        # integrate pdf over the bin        
        prob[i], _ = quadgk(f, x_min, x_max)                  
    end

    prob /= integral_of_f  # scale by integral of f to get normalised probabilities

    return prob
end


# function compute_2D_mean_L1_error(q_traj, probabilities, x_bins, y_bins, n_bins)
#     # Compute the mean L1 error between the expected probability and
#     # the actual probability distributions across all bins of a 2D histogram

#     # Compute the histogram
#     h2 = Hist2D(collect(eachrow(q_traj)), (x_bins, y_bins))

#     # Compute the means L1 error between the expected probability and the actual probability distributions
#     mean_error = sum(abs.(bincounts(h2)/length(q_traj[1, :]) .- probabilities)) / (n_bins * n_bins)

#     return h2, mean_error
# end

function compute_1D_mean_L1_error(empirical_histogram::Hist1D, theoretical_probabilities::Vector{Float64})
    # Compute the mean L1 error between the expected probability and
    # the actual probability distributions across all bins of a 1D histogram

    n_samples = integral(empirical_histogram)
    n_bins = nbins(empirical_histogram)

    # Compute the means L1 error between the expected probability and the actual probability distributions
    mean_error = sum(abs.(bincounts(empirical_histogram)/n_samples .- theoretical_probabilities)) / (n_bins)

    return mean_error
end

function compute_1D_mean_L1_error(empirical_probabilities::Vector{Float64}, theoretical_probabilities::Vector{Float64})
    # Compute the mean L1 error between the expected probability and
    # the actual probability distributions across all bins of a 1D histogram

    n_bins = length(empirical_probabilities)

    # Compute the means L1 error between the expected probability and the actual probability distributions
    mean_error = sum(abs.(empirical_probabilities .- theoretical_probabilities)) / (n_bins)

    return mean_error
end

function compute_2D_mean_L1_error(empirical_histogram::Hist2D, theoretical_probabilities::Matrix{Float64})
    # Compute the mean L1 error between the expected probability and
    # the actual probability distributions across all bins of a 2D histogram

    n_samples = integral(empirical_histogram)
    n_bins_x, n_bins_y = nbins(empirical_histogram)

    # Compute the means L1 error between the expected probability and the actual probability distributions
    mean_error = sum(abs.(bincounts(empirical_histogram)/n_samples .- theoretical_probabilities)) / (n_bins_x * n_bins_y)

    return mean_error
end


function compute_2D_mean_L1_error(empirical_probabilities::Matrix{Float64}, theoretical_probabilities::Matrix{Float64})
    # Compute the mean L1 error between the expected probability and
    # the actual probability distributions across all bins of a 2D histogram

    n_bins_x, n_bins_y = size(empirical_probabilities)

    # Compute the mean L1 error between the expected probability and the actual probability distributions
    mean_error = sum(abs.(empirical_probabilities .- theoretical_probabilities)) / (n_bins_x * n_bins_y)

    return mean_error
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

function plot_histograms(integrator, histogram_data, stepsizes, save_dir; xlabel="x", ylabel="y", title="2D Histogram")
    for (stepsize_idx, dt) in enumerate(stepsizes)
        for repeat in 1:size(histogram_data, 2)
            plot_heatmap(histogram_data[stepsize_idx, repeat], repeat, save_dir, integrator, dt, xlabel=xlabel, ylabel=ylabel, title=title)
        end
    end
end

function init_q0(q0; dim::Int = 1) 
    if q0 === nothing
        q0 = randn(dim)
    end
    return q0
end

function plot_heatmap(hist::Hist2D, repeat, save_dir, integrator, dt; xlabel, ylabel, title)
    # Extract histogram data
    bins_x = bincenters(hist)[1]
    bins_y = bincenters(hist)[2]
    freq = bincounts(hist)
    
    # Plot heatmap
    h = heatmap(bins_x, bins_y, freq, aspect_ratio=:equal, color=:viridis,
            xlabel=xlabel, ylabel=ylabel, title=title)

    # Save heatmap
    savefig(h, "$(save_dir)/heatmaps/$(string(nameof(integrator)))/h=$dt/$(repeat).png")
end

end # module Utils