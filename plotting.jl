module Plotting
using Plots
export plot_trajectory, plot_2D_potential

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

end # module Plotting