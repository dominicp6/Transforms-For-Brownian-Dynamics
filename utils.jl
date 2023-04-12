module Utils
using HCubature, QuadGK, FHist
export compute_1D_probabilities, compute_2D_probabilities, compute_1D_mean_L1_error, compute_2D_mean_L1_error

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


function compute_2D_mean_L1_error(q_traj, probabilities, x_bins, y_bins, n_bins)
    # Compute the mean L1 error between the expected probability and
    # the actual probability distributions across all bins of a 2D histogram

    # Compute the histogram
    h2 = Hist2D(collect(eachrow(q_traj)), (x_bins, y_bins))

    # Compute the means L1 error between the expected probability and the actual probability distributions
    mean_error = sum(abs.(bincounts(h2)/length(q_traj[1, :]) .- probabilities)) / (n_bins * n_bins)

    return h2, mean_error
end


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

# function debias_experiment(exp_dir, transform)
#     # Create a duplicate debiased experiment directory and debias the histograms in the checkpoints directory
#     # transform must map back to the desired range

#     # check if the checkpoints directory exists inside exp_dir
#     checkpoints_dir = joinpath(exp_dir, "checkpoints")
#     if !isdir(checkpoints_dir)
#         error("No checkpoints directory found in $exp_dir")
#     end

#     integrator_names = []

#     for integrator_folder in readdir(checkpoints_dir)
#         # check if the subfolder is a directory
#         if isdir(joinpath(checkpoints_dir, integrator_folder))
           
#             # make a duplicate folder under the debiased directory
#             debiased_dir = joinpath(exp_dir, "debiased")
#             if !isdir(debiased_dir)
#                 mkdir(debiased_dir)
#             end
#             debiased_integrator_dir = joinpath(debiased_dir, integrator_folder)
#             if !isdir(debiased_integrator_dir)
#                 mkdir(debiased_integrator_dir)
#             end

#             # append the name of the directory to integrator names
#             push!(integrator_names, integrator_folder)
#             # get the path to the integrator directory
#             integrator_dir = joinpath(checkpoints_dir, integrator_folder)
#             for stepsize_exp in readdir(integrator_dir)
#                 if isdir(joinpath(integrator_dir, stepsize_exp))
                    
#                     # make a duplicate folder under the debiased directory
#                     debiased_stepsize_exp_dir = joinpath(debiased_integrator_dir, stepsize_exp)
#                     if !isdir(debiased_stepsize_exp_dir)
#                         mkdir(debiased_stepsize_exp_dir)
#                     end

#                     # read in each file in the directory as a separate histogram
#                     stepsize_exp_dir = joinpath(integrator_dir, stepsize_exp)
#                     for file in readdir(stepsize_exp_dir)
#                         if endswith(file, ".jld2")
#                             opened_file = jldopen(joinpath(stepsize_exp_dir, file), "r")
#                             hist = read(opened_file, "data")
#                             # transform the bin edges of the histogram
#                             debiased_hist = debias_hist(hist, transform)
                            
#                             # save the debiased histogram to the debiased directory
#                             file = replace(file, ".jld2" => "")
#                             path_to_file = joinpath(debiased_stepsize_exp_dir, file)
#                             h5write(string(path_to_file,"_","centres.h5"), "data", bincenters(hist))
#                             h5write(string(path_to_file,"_","counts.h5"), "data", bincounts(hist))
#                             #save(joinpath(debiased_stepsize_exp_dir, file), "data", debiased_hist)
#                         end
#                     end
#                 end
#             end
#         end
#     end
# end

# function generate_convergence_plots_from_checkpoints(checkpoint_dir, theoretical_probabilities)
#     convergence_data = nothing
#     stepsizes = nothing

#     # loop through all the integrators and stepsize experiments and generate convergence plots
#     for integrator_folder in readdir(checkpoint_dir)
#         # check if the subfolder is a directory
#         if isdir(joinpath(checkpoint_dir, integrator_folder))
#             # get the path to the integrator directory
#             integrator_dir = joinpath(checkpoint_dir, integrator_folder)
#             if stepsizes === nothing
#                 # extract the step sizes from the trailing numeric part of the trajectory names
#                 stepsizes = [parse(Float64, match(r"\d+\.\d+", stepsize_exp).match) for stepsize_exp in readdir(integrator_dir)]
#             end
#             # get number of step_sizes in the integrator directory
#             n_step_sizes = length(readdir(integrator_dir))

#             for (step_idx, stepsize_exp) in enumerate(readdir(integrator_dir))
#                 if isdir(joinpath(integrator_dir, stepsize_exp))
#                     # read in each file in the directory as a separate histogram
#                     stepsize_exp_dir = joinpath(integrator_dir, stepsize_exp)
#                     # get number of repeats
#                     n_repeats = length(readdir(stepsize_exp_dir))
#                     if convergence_data === nothing
#                         convergence_data = zeros(n_step_sizes, n_repeats)
#                     end
#                     for (repeat_idx, file) in enumerate(readdir(stepsize_exp_dir))
#                         if endswith(file, ".jld2")
#                             opened_file = jldopen(joinpath(stepsize_exp_dir, file), "r")
#                             hist = read(opened_file, "data")
#                             # compute the mean L1 error
#                             mean_error = compute_1D_mean_L1_error(hist, theoretical_probabilities)
#                             # save the mean error to the convergence data
#                             convergence_data[step_idx, repeat_idx] = mean_error
#                         end
#                     end
#                 end
#             end
#             # save the convergence data for that integrator
#             save(joinpath(checkpoint_dir, integrator_folder, "convergence_data.jld2"), "data", convergence_data)

#             # Plot (dim 1 is the step size, dim 2 is the repeat)
#             plot(stepsizes, mean(convergence_data, dims=2), title=string(integrator_folder), xlabel="dt", ylabel="Mean L1 error", xscale=:log10, yscale=:log10, label="")
#             plot!(stepsizes, mean(convergence_data, dims=2)+std(convergence_data, dims=2), ls=:dash, lc=:black, label="")
#             plot!(stepsizes, mean(convergence_data, dims=2)-std(convergence_data, dims=2), ls=:dash, lc=:black, label="")
#             # save the plot
#             savefig(joinpath(checkpoint_dir, integrator_folder, "convergence_plot.png"))
#         end
#     end
# end

end # module Utils