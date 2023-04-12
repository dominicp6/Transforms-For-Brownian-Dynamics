module TransformUtils
using FHist, StatsBase
export increment_g_counts, time_transformed_potential

function time_transformed_potential(x, V, D, tau)
    return V(x) - tau * log(D(x))
end

function scale_range(range, transform)
    return transform.(range)
end

function debias_hist(hist, transform)
    # De-bias a histogram by transforming the bin edges according to a given function
    bin_edges = binedges(hist)
    new_hist_edges = scale_range(bin_edges, transform)
    debias_hist = Hist1D(Histogram(new_hist_edges, bincounts(hist)))

    return debias_hist
end

function increment_g_counts(q_chunk, D, bin_boundaries, ΣgI, Σg)
    g(x) = 1/D(x)
    
    # Iterate through trajectory points and assign to corresponding bin
    for q in q_chunk
        Σg += g(q)

        # Find the index of the histogram bin that q is in
        bin_index = searchsortedfirst(bin_boundaries, q) - 1
        # only count points that are in the domain of the specified bins
        if bin_index != 0 && bin_index != length(bin_boundaries)
            ΣgI[bin_index] += g(q)
        end
    end

    return ΣgI, Σg
end

function average_renormalised_probs(renormalised_probs, chunk_size, steps_ran)
    num_chunks = length(renormalised_probs)

    # Compute the cumulative sum of probabilities
    cumsum = zeros(size(renormalised_probs[1]))
    for i in 1:num_chunks-1
        cumsum .+= renormalised_probs[i] .* chunk_size
    end
    cumsum .+= renormalised_probs[end] .* steps_ran

    # Compute the total number of steps and return the average
    total_steps = chunk_size * (num_chunks - 1) + steps_ran
    return cumsum ./ total_steps
end

end # module