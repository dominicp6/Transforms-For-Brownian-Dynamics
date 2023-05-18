module Integrators
include("calculus.jl")
using LinearAlgebra, Random, Plots, ForwardDiff, Base.Threads, ProgressBars, .Calculus
export euler_maruyamaND, naive_leimkuhler_matthewsND, hummer_leimkuhler_matthewsND, euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D

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

function milstein_method1D(q0, Vprime, D, Dprime, tau::Number, m::Integer, dt::Number)
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

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        Dq = D(q)
        grad_V = Vprime(q)
        grad_D = Dprime(q)
        drift = -Dq * grad_V + tau * grad_D
        Rₖ = randn()
        diffusion = sqrt(2 * tau * Dq) * Rₖ
        second_order_correction = (tau/2) * grad_D * (Rₖ^2 - 1)
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) + second_order_correction * dt
        q_traj[i] = q
        
        # update the time
        t += dt   
    end 
    
    return q_traj
end

function stochastic_heun1D(q0, Vprime, D, Dprime, tau::Number, m::Integer, dt::Number)
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
        # Compute drift and diffusion coefficients at the current position
        Dq = D(q)
        grad_V = Vprime(q)
        grad_D = Dprime(q)
        drift = -Dq * grad_V + 0.5 * tau * grad_D   #  gradient term gets 0.5 factor to correct for Stratanovich interpretation
        diffusion = sqrt(2 * tau * Dq)
        
        # Compute the predicted next state using Euler-Maruyama
        Rₖ = randn()
        q_pred = q + drift * dt + diffusion * Rₖ * sqrt(dt)
        
        # Compute drift and diffusion coefficients at the predicted position
        Dq_pred = D(q_pred)
        grad_V_pred = Vprime(q_pred)
        grad_D_pred = Dprime(q_pred)
        drift_pred = -Dq_pred * grad_V_pred + 0.5 * tau * grad_D_pred   # gradient term gets 0.5 factor to correct for Stratanovich interpretation
        diffusion_pred = sqrt(2 * tau * Dq_pred)
        
        # Compute the corrected next state using a weighted average
        q += 0.5 * (drift + drift_pred) * dt + 0.5 * (diffusion + diffusion_pred) * Rₖ * sqrt(dt)
        q_traj[i] = q
        
        # Update the time
        t += dt
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

function naive_leimkuhler_matthewsND_efficient(q0, Vprime, DDT, div_DDT, tau::Number, m::Integer, dt::Number)
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
        grad_V = Vprime(q)
        DDTq = DDT(q)
        div_DDTq = div_DDT(q)
        drift = -DDTq * grad_V + tau * div_DDTq 
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