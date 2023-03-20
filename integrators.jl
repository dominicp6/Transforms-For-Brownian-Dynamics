module Integrators
include("calculus.jl")
using LinearAlgebra, Random, Plots, ForwardDiff, Base.Threads, ProgressBars, .Calculus
export euler_maruyama, naive_leimkuhler_matthews, hummer_leimkuhler_matthews, euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D

function euler_maruyama1D(q0, V, D, tau, T, dt, seed)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion coefficient at a given configuration
    # tau is the temperature parameter
    # T is the final time
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    m = ceil(Int, T/dt)
    q_traj = zeros(m+1)
    q_traj[1] = q
    Random.seed!(seed) # set the random seed for reproducibility

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = ForwardDiff.derivative(V, q)
        grad_D = ForwardDiff.derivative(D, q)
        drift = -Dq * grad_V + tau * grad_D
        diffusion = sqrt(2 * tau * Dq) * randn()
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt)
        q_traj[i+1] = q
        
        # update the time
        t += dt
    end
    
    return q_traj
end

function naive_leimkuhler_matthews1D(q0, V, D, tau, T, dt, seed)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion coefficient at a given configuration
    # tau is the time scale for diffusion
    # T is the final time
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    m = ceil(Int, T/dt)
    q_traj = zeros(m+1)
    q_traj[1] = q
    Random.seed!(seed) # set the random seed for reproducibility
    Rₖ = randn()

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = ForwardDiff.derivative(V, q)
        grad_D = ForwardDiff.derivative(D, q)
        drift = -Dq * grad_V + tau * grad_D
        Rₖ₊₁ = randn()
        diffusion = sqrt(2 * tau * Dq) * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[i+1] = q
        
        # update the time
        t += dt

        # update the last increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj
end

function hummer_leimkuhler_matthews1D(q0, V, D, tau, T, dt, seed)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion coefficient at a given configuration
    # tau is the time scale for diffusion
    # T is the final time
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    m = ceil(Int, T/dt)
    q_traj = zeros(m+1)
    q_traj[1] = q
    Random.seed!(seed) # set the random seed for reproducibility
    Rₖ = randn()

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = ForwardDiff.derivative(V, q)
        grad_D = ForwardDiff.derivative(D, q)
        drift = -Dq * grad_V + tau * grad_D
        Rₖ₊₁ = randn()
        diffusion = sqrt(2 * tau * Dq) * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[i+1] = q
        
        # update the time
        t += dt

        # update the last increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj
end

function euler_maruyama(q0, V, D, tau, T, dt, seed)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion tensor at a given configuration
    # tau is the temperature parameter
    # T is the final time
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    m = ceil(Int, T/dt)
    q_traj = zeros(n, m+1)
    q_traj[:,1] .= q
    Random.seed!(seed) # set the random seed for reproducibility

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
        q_traj[:,i+1] .= q
        
        # update the time
        t += dt
    end
    
    return q_traj
end

function naive_leimkuhler_matthews(q0, V, D, tau, T, dt, seed)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion tensor at a given configuration
    # tau is the time scale for diffusion
    # T is the final time
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    m = ceil(Int, T/dt)
    q_traj = zeros(n, m+1)
    q_traj[:,1] .= q
    Random.seed!(seed) # set the random seed for reproducibility
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
        q_traj[:,i+1] .= q
        
        # update the time
        t += dt

        # update the last increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj
end

function hummer_leimkuhler_matthews(q0, V, D, tau, T, dt, seed)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion tensor at a given configuration
    # tau is the time scale for diffusion
    # T is the final time
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    m = ceil(Int, T/dt)
    q_traj = zeros(n, m+1)
    q_traj[:,1] .= q
    Random.seed!(seed) # set the random seed for reproducibility
    Rₖ = randn(n)

    # simulate
    for i in ProgressBar(1:m)
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = ForwardDiff.gradient(V, q)
        DDT = Dq * Dq'
        div_DDT = matrix_divergence(x -> D(x) * D(x)', q)
        drift = -DDT * grad_V + (3/4)*tau * div_DDT 
        Rₖ₊₁ = randn(n)
        diffusion = sqrt(2 * tau) * Dq * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[:,i+1] .= q
        
        # update the time
        t += dt

        # update the last increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj
end

end # module Integrators