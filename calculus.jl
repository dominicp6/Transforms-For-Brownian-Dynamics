module Calculus
using ForwardDiff, Symbolics
export matrix_divergence, differentiate1D

function differentiate1D(f)
    # Compute the symbolic derivative of a scalar function of a single variable
    @variables x
    gradGen = Differential(x)(f(x))
    gradExp = expand_derivatives(gradGen)
    gradFn = Symbolics.build_function(gradExp,x, expression=false)

    return gradFn
end

function vector_divergence(V, q)
    # Compute the divergence of a vector function evaluated at vector variable q
    div = 0
    for i in eachindex(q)
        div += ForwardDiff.gradient(x -> V(x)[i], q)[i]
    end
    return div
end

function matrix_divergence(M, q)
    # Compute the matrix divergence of a matrix function evaluated at vector variable q
    # The matrix divergence is defined as the column vector resulting from the vector divergence of each row
    div_M = zeros(size(q))
    for i in eachindex(q)
        # Take the divergence of the ith row of the matrix function
        div_M[i] = vector_divergence(x -> M(x)[i,:], q)
    end
    return div_M
end

end # module Calculus