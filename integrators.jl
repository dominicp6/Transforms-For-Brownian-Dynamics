module Integrators
include("calculus.jl")
using LinearAlgebra, Random, Plots, ForwardDiff, Base.Threads, ProgressBars, .Calculus
export euler_maruyamaND, naive_leimkuhler_matthewsND, hummer_leimkuhler_matthewsND, euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D

function euler_maruyama1D(q0, Vprime, D, Dprime, tau::Number, m::Integer, dt::Number)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion coefficient at a given configuration
    # tau is the temperature parameter
    # m is the number of steps
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    q_traj = zeros(m)

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = Vprime(q)
        grad_D = Dprime(q)
        drift = -Dq * grad_V + tau * grad_D
        diffusion = sqrt(2 * tau * Dq) * randn()
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt)
        q_traj[i] = q
        
        # update the time
        t += dt
    end
    
    return q_traj
end

function naive_leimkuhler_matthews1D(q0, Vprime, D, Dprime, tau::Number, m::Integer, dt::Number)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion coefficient at a given configuration
    # tau is the time scale for diffusion
    # m is the number of steps
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    q_traj = zeros(m)
    Rₖ = randn()

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = Vprime(q)
        grad_D = Dprime(q)
        drift = -Dq * grad_V + tau * grad_D
        Rₖ₊₁ = randn()
        diffusion = sqrt(2 * tau * Dq) * (Rₖ + Rₖ₊₁)/2 
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[i] = q
        
        # update the time
        t += dt

        # update the last increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj
end

function hummer_leimkuhler_matthews1D(q0, Vprime, D, Dprime, tau::Number, m::Integer, dt::Number)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion coefficient at a given configuration
    # tau is the time scale for diffusion
    # m is the number of steps
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    q_traj = zeros(m)
    Rₖ = randn()

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = Vprime(q)
        grad_D = Dprime(q)
        drift = -Dq * grad_V + (3/4) * tau * grad_D
        Rₖ₊₁ = randn()
        diffusion = sqrt(2 * tau * Dq) * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[i] = q
        
        # update the time
        t += dt

        # update the last increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj
end

function euler_maruyamaND(q0, V, D, tau::Number, m::Integer, dt::Number)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion tensor at a given configuration
    # tau is the temperature parameter
    # m is the number of steps
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    q_traj = zeros(n, m)

    # simulate
    for i in ProgressBar(1:m)
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = ForwardDiff.gradient(V, q)
        DDT = Dq * Dq'
        div_DDT = matrix_divergence(x -> D(x) * D(x)', q)
        drift = -DDT * grad_V + tau * div_DDT 
        diffusion = sqrt(2 * tau) * Dq * randn(n)
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt)
        q_traj[:,i] .= q
        
        # update the time
        t += dt
    end
    
    return q_traj
end

function naive_leimkuhler_matthewsND(q0, V, D, tau::Number, m::Integer, dt::Number)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion tensor at a given configuration
    # tau is the time scale for diffusion
    # m is the number of steps
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    q_traj = zeros(n, m)
    Rₖ = randn(n)

    # simulate
    for i in ProgressBar(1:m)
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = ForwardDiff.gradient(V, q)
        DDT = Dq * Dq'
        div_DDT = matrix_divergence(x -> D(x) * D(x)', q)
        drift = -DDT * grad_V + tau * div_DDT 
        Rₖ₊₁ = randn(n)
        diffusion = sqrt(2 * tau) * Dq * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[:,i] .= q
        
        # update the time
        t += dt

        # update the last increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj
end

function hummer_leimkuhler_matthewsND(q0, V, D, tau::Number, m::Integer, dt::Number)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion tensor at a given configuration
    # tau is the time scale for diffusion
    # m is the number of steps
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    q_traj = zeros(n, m)
    Rₖ = randn(n)

    # simulate
    for i in ProgressBar(1:m)
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = ForwardDiff.gradient(V, q)
        DDT = Dq * Dq'
        div_DDT = matrix_divergence(x -> D(x) * D(x)', q)
        drift = -DDT * grad_V + (3/4) * tau * div_DDT 
        Rₖ₊₁ = randn(n)
        diffusion = sqrt(2 * tau) * Dq * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[:,i] .= q
        
        # update the time
        t += dt

        # update the last increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj
end

end # module Integrators