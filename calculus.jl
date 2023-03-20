module Calculus
using ForwardDiff
export matrix_divergence

function vector_divergence(V, q)
    # compute the divergence of a vector field
    div = 0
    for i in eachindex(q)
        div += ForwardDiff.gradient(x -> V(x)[i], q)[i]
    end
    return div
end


function matrix_divergence(M, q)
    # compute the divergence of the diffusion tensor
    # we define the divergence of a matrix as the column vector resulting from the divergence of each row
    div_M = zeros(size(q))
    for i in eachindex(q)
        # take the divergence of the ith row of the diffusion tensor
        div_M[i] = vector_divergence(x -> M(x)[i,:], q)
    end
    return div_M
end

end # module Calculus