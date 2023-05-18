module Calculus
using ForwardDiff, Symbolics
export matrix_divergence, differentiate1D, differentiateND, differentiate2D, symbolic_vector_divergence2D

function differentiate1D(f)
    # Compute the symbolic derivative of a scalar function of a single variable
    @variables x
    gradGen = Differential(x)(f(x))
    gradExp = expand_derivatives(gradGen)
    gradFn = Symbolics.build_function(gradExp,x, expression=false)

    return gradFn
end

function differentiate2D(f::Function)
    @variables x y
    grad_f_expr = Symbolics.gradient(f(x,y), [x,y])
    grad_f_func = build_function(grad_f_expr, [x,y])
    return grad_f_func
end

function symbolic_vector_divergence2D(V)
    # Compute the symbolic divergence of a vector function
    div = 0
    @variables x y
    div1 = Differential(x)(V(x,y))[1]
    div2 = Differential(y)(V(x,y))[2]
    div1 = expand_derivatives(div)
    div2 = expand_derivatives(div)
    div1Fn = Symbolics.build_function(div1, [x,y], expression=false)
    println(div1Fn(1.0,1.0))
    div2Fn = Symbolics.build_function(div2, [x,y], expression=false)
    println(div2Fn(1.0,1.0))
    divFn(x, y) = div1Fn(x, y) + div2Fn(x, y)
    println(divFn(1.0,1.0))

    # div1 = Symbolics.derivative(V, [x,y], 1)
    # div1 = expand_derivatives(div1)
    # div1Fn = Symbolics.build_function(div1, [x,y], expression=false)
    # div2 = Symbolics.derivative(V, [x,y], 2)
    # div2 = expand_derivatives(div2)
    # div2Fn = Symbolics.build_function(div2, [x,y], expression=false)
    # println(div1Fn(1.0,1.0))    
    # println(div2Fn(1.0,1.0))
    return div
end

function symbolic_matrix_divergence2D(M)
    # Compute the symbolic matrix divergence of a matrix function
    # The matrix divergence is defined as the column vector resulting from the vector divergence of each row
    div_M = zeros(2)
    # Take the divergence of the ith row of the matrix function
    div_M[1] = symbolic_vector_divergence2D((x,y) -> M(x,y)[1,:])
    div_M[2] = symbolic_vector_divergence2D((x,y) -> M(x,y)[2,:])
    return div_M
end

# function differentiate2D(f::Function)
#     @variables x y
#     grad_x = differentiate1D((x_val) -> f(x_val, y))
#     grad_y = differentiate1D((y_val) -> f(x, y_val))
#     gradFn = [grad_x(x), grad_y(y)]
#     return gradFn
# end

# function differentiate2D(f::Function)
#     x, y = @variables x y
#     gradGen = Symbolics.gradient(f([x,y]), [x, y])
#     gradFn = build_function(gradGen, [x, y], expression=false)
#     return gradFn
# end

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