module DiffusionTensors
using LinearAlgebra
export Dconst1D, Dabs1D, Dquadratic1D, Dconst2D, Dlinear2D, Dquadratic2D, DmoroCardin, Doseen

function Dconst1D(q::T) where T<:Real
    return 1.0
end

function Dabs1D(q::T) where T<:Real
    return 1.0 + abs(q)
end
    
function Dquadratic1D(q::T) where T<:Real
    return 1.0 + q^2 
end

function Dconst2D(x::T, y::T) where T<:Real
    return Matrix{Float64}(I, 2, 2)
end

function Dlinear2D(q::AbstractVector{T}) where T<:Real
    x, y = q
    return (abs(x) + abs(y) + 0.001) * Matrix{Float64}(I, 2, 2)
end

function Dquadratic2D(x::T, y::T) where T<:Real
    return (x^2 + y^2 + 0.001) * Matrix{Float64}(I, 2, 2)
end

function DmoroCardin(x::T, y::T) where T<:Real
    return (1.0 + 5.0 * exp(- (x^2 + y^2) / (2 * 0.3^2)))^(-1) * Matrix{Float64}(I, 2, 2)
end

function Doseen(q::AbstractVector{T}) where T<:Real
    x, y = q
    r2 = x^2 + y^2
    return [1.0 + x^2/r2 x*y/r2; x*y/r2 1 + y^2/r2]
end

end # module DiffusionTensors