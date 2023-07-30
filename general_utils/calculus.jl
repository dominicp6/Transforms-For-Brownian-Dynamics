module Calculus
using ForwardDiff, Symbolics
export matrix_divergence, differentiate1D, differentiateND, differentiate2D, symbolic_vector_divergence2D

function differentiate1D(f)
    # Compute the symbolic derivative of a scalar function of a single variable
    @variables x
    gradGen = Differential(x)(f(x))
    gradExp = expand_derivatives(gradGen)
    gradFn = Symbolics.build_function(gradExp, x, expression=false)

    return gradFn
end

function differentiate2D(f::Function)
    # Compute the symbolic gradient of a scalar function of two variables
    @variables x y
    grad_f_expr = Symbolics.gradient(f(x,y), [x,y])
    grad1 = build_function(grad_f_expr[1], [x,y], expression=false)
    grad2 = build_function(grad_f_expr[2], [x,y], expression=false)
    grad = (x, y) -> begin
        [grad1([x, y]), grad2([x, y])]
    end

    return grad
end

function symbolic_vector_divergence2D(V)
    # Compute the symbolic divergence of a vector function
    @variables x y
    div1 = Differential(x)(V(x,y)[1])
    div2 = Differential(y)(V(x,y)[2])
    div1 = expand_derivatives(div1)
    div2 = expand_derivatives(div2)
    div1Fn = Symbolics.build_function(div1, [x,y], expression=false)
    div2Fn = Symbolics.build_function(div2, [x,y], expression=false)
    div = (x, y) -> begin
        div1Fn([x, y]) + div2Fn([x, y])
    end

    return div   #println(div(1.0, 2.0)) # expect 17.0
end

function symbolic_matrix_divergence2D(M)
    # Compute the symbolic matrix divergence of a matrix function
    # The matrix divergence is defined as the column vector resulting from the vector divergence of each row
    V1 = (x, y) -> M(x,y)[1,:]
    V2 = (x, y) -> M(x,y)[2,:]
    div_M1 = symbolic_vector_divergence2D(V1)
    div_M2 = symbolic_vector_divergence2D(V2)
    div_M = (x, y) -> begin
        [div_M1(x, y),  div_M2(x, y)]
    end
    
    return div_M
end

function differentiateND(f::Function)
    # Compute the symbolic gradient of a scalar function of multiple variables
    @variables x[1:length(f.args)...]
    gradGen = gradient(f(x...), x)
    gradFn = build_function(gradGen, x, expression=false)

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