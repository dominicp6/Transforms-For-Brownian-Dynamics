module Potentials
export bowl2D, doubleWell1D, quadrupleWell2D, moroCardin2D, muller_brown, LM2013

function bowl2D(q::AbstractVector{T}) where T<:Real
    # 2D bowl potential
    x, y = q
    0.5*(x^2+y^2)
end


function doubleWell1D(q::T) where T<:Real
    # 1D double well potential
    h = 2
    c = 2
    return -(1/4)*(q^2)*(h^4) + (1/2)*(c^2)*(q^4)
end


function LM2013(q::T) where T<:Real
    # 1D potential from L. M. 2013
    return q^4 /4 + sin(1 + 5q)
end


function quadrupleWell2D(q::AbstractVector{T}) where T<:Real
    # 2D quadruple well potential
    x, y = q
    h = 2
    c = 2
    return -(1/4)*(x^2)*(h^4) + (1/2)*(c^2)*(x^4) + -(1/4)*(y^2)*(h^4) + (1/2)*(c^2)*(y^4)
end


function moroCardin2D(q::AbstractVector{T}) where T<:Real
    # 2D Moro-Cardin potential
    x, y = q
    return 5*(x^2-1)^2 + 10*atan(7*pi/9)*y^2
end

function muller_brown(q::AbstractVector{T}) where T <: Real
    A = [-200.0; -100.0; -170.0; 15.0]
    a = [-1.0; -1.0; -6.5; 0.7]
    b = [0.0; 0.0; 11.0; 0.6]
    c = [-10.; -10.; -6.5; 0.7]
    x_ = [1.; 0.; -0.5; -1.] 
    y_ = [0.; 0.5; 1.5; 1.]

    z = sum(A .* exp.(a .* (q[1] .- x_).^2 .+ b .* (q[1] .- x_) .* (q[2] .- y_) .+ c .* (q[2] .- y_).^2))
    
    return z
end

end # module Potentials