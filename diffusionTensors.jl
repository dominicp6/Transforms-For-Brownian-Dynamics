module DiffusionTensors
using LinearAlgebra
export Dconst1D, Dlinear1D, Dquadratic1D, Dconst2D, Dlinear2D, Dquadratic2D, DmoroCardin, Doseen

function Dconst1D(q::T) where T<:Real
    return 1.0
end

function Dlinear1D(q::T) where T<:Real
    return abs(q) + 0.001
end
    
function Dquadratic1D(q::T) where T<:Real
    return q^2 + 0.001
end

function Dconst2D(q::AbstractVector{T}) where T<:Real
    return Matrix{Float64}(I, 2, 2)
end

function Dlinear2D(q::AbstractVector{T}) where T<:Real
    x, y = q
    return (abs(x) + abs(y) + 0.001) * Matrix{Float64}(I, 2, 2)
end

function Dquadratic2D(q::AbstractVector{T}) where T<:Real
    x, y = q
    return (x^2 + y^2 + 0.001) * Matrix{Float64}(I, 2, 2)
end

function DmoroCardin(q::AbstractVector{T}) where T<:Real
    x, y = q
    return (1 + 8 * exp(- (x^2 + y^2) / (2 * 0.2^2)))^(-1) * Matrix{Float64}(I, 2, 2)
end

function Doseen(q::AbstractVector{T}) where T<:Real
    x, y = q
    r2 = x^2 + y^2
    return [1 + x^2/r2 x*y/r2; x*y/r2 1 + y^2/r2]
end

end # module DiffusionTensors