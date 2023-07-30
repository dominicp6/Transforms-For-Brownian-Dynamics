module PlottingUtils
using Plots, FHist
export plot_trajectory, plot_2D_potential, plot_histograms, save_and_plot

function plot_trajectory(q, dt)
    n = size(q, 1)
    t = collect(0:(size(q, 2)-1)) * dt
    p = plot(t, q[1,:], xlabel="Time", ylabel="x", legend=false)
    for i in 2:n
        plot!(p, t, q[i,:], xlabel="Time", ylabel="y", legend=false)
    end
    return p
end

function plot_2D_potential(potential, xmin, ymin, xmax, ymax)
    length = 100
    x = range(xmin, xmax, length=length)
    y = range(ymin, ymax, length=length)

    z = @. potential([x', y])

    return contour(x, y, z, st=:surface, xlabel="x", ylabel="y", zlabel="V(x,y)", legend=false)
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

function plot_histograms(integrator, histogram_data, stepsizes, save_dir; xlabel="x", ylabel="y", title="2D Histogram")
    for (stepsize_idx, dt) in enumerate(stepsizes)
        for repeat in 1:size(histogram_data, 2)
            plot_heatmap(histogram_data[stepsize_idx, repeat], repeat, save_dir, integrator, dt, xlabel=xlabel, ylabel=ylabel, title=title)
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

function save_and_plot(integrator, convergence_data, diffusion_coefficient_data, stepsizes, save_dir; xlabel="dt", ylabel="Mean L1 error", error_in_mean=false)
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

    # Plot the diffusion coefficient
    plot(diffusion_coefficient_data[1,1][1], diffusion_coefficient_data[1,1][2], title=string(nameof(integrator)), xlabel="x", ylabel="Diffusion coefficient")
    savefig("$(save_dir)/$(integrator)_diffusion.png")
end

end # module PlottingUtils