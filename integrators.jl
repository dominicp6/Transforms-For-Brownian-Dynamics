module Integrators
include("calculus.jl")
using LinearAlgebra, Random, Plots, ForwardDiff, Base.Threads, ProgressBars
using .Calculus: symbolic_matrix_divergence2D
export euler_maruyama1D, naive_leimkuhler_matthews1D, hummer_leimkuhler_matthews1D, milstein_method1D, stochastic_heun1D, euler_maruyama2D, naive_leimkuhler_matthews2D, hummer_leimkuhler_matthews2D, euler_maruyama2D_identityD, naive_leimkuhler_matthews2D_identityD, leimkuhler_matthews1D, leimkuhler_matthews2D

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

        # update the noise increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj
end

function leimkuhler_matthews1D(q0, Vprime, D, Dprime, tau::Number, m::Integer, dt::Number)
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

    sqrtD = x -> sqrt(D(x))

    # number of inner loop steps
    n = 5

    # tau = 1 in the following
    # simulate
    for i in 1:m
        sqrtDq = sqrtD(q)
        grad_V = Vprime(q)
        grad_D = Dprime(q)
        hat_pₖ₊₁ = sqrt(tau) * Rₖ - sqrt(2 * dt) * sqrtDq * grad_V + sqrt(dt / 2) * grad_D / sqrtDq
        
        sqrt_h_2 = sqrt(dt / 2)
        inner_step = sqrt_h_2 / n  # Divide by n for each internal RK4 step
        
        # Perform n steps of RK4 integration for hat_qₖ₊₁
        hat_qₖ₊₁ = q # Initialize hat_qₖ₊₁
        for j in 1:n
            # Compute intermediate values
            k1 = inner_step * sqrtD(hat_qₖ₊₁) * hat_pₖ₊₁
            k2 = inner_step * sqrtD(hat_qₖ₊₁ + 0.5 * k1) * hat_pₖ₊₁
            k3 = inner_step * sqrtD(hat_qₖ₊₁ + 0.5 * k2) * hat_pₖ₊₁
            k4 = inner_step * sqrtD(hat_qₖ₊₁ + k3) * hat_pₖ₊₁
            
            # Update state using weighted average of intermediate values
            hat_qₖ₊₁ += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        end
        
        # Perform n steps of RK4 integration for qₖ₊₁
        qₖ₊₁ = hat_qₖ₊₁ # Initialize qₖ₊₁
        Rₖ₊₁ = randn()
        for j in 1:n
            # Compute intermediate values
            k1 = inner_step * sqrtD(qₖ₊₁) * Rₖ₊₁
            k2 = inner_step * sqrtD(qₖ₊₁ + 0.5 * k1) * Rₖ₊₁
            k3 = inner_step * sqrtD(qₖ₊₁ + 0.5 * k2) * Rₖ₊₁
            k4 = inner_step * sqrtD(qₖ₊₁ + k3) * Rₖ₊₁
            
            # Update state using weighted average of intermediate values
            qₖ₊₁ += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * sqrt(tau)
        end
        
        # Update the trajectory
        q = qₖ₊₁
        q_traj[i] = q
        
        # update the time
        t += dt

        # update the noise increment
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

        # update the noise increment
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

function euler_maruyama2D(q0, Vprime, D, div_DDT, tau::Number, m::Integer, dt::Number)
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
    for i in 1:m
        # compute the drift and diffusion coefficients
        grad_V = Vprime(q[1], q[2])
        Dq = D(q[1], q[2])
        DDTq = Dq * Dq'
        div_DDTq = div_DDT(q[1], q[2])
        drift = -DDTq * grad_V + tau * div_DDTq 
        diffusion = sqrt(2 * tau) * Dq * randn(n)
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt)
        q_traj[:,i] .= q
        
        # update the time
        t += dt
    end
    
    return q_traj
end

function euler_maruyama2D_identityD(q0, Vprime, D, div_DDT, tau::Number, m::Integer, dt::Number)
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
    for i in 1:m
        # compute the drift and diffusion coefficients
        grad_V = Vprime(q[1], q[2])
        drift = -grad_V
        diffusion = sqrt(2 * tau) * randn(n)
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt)
        q_traj[:,i] .= q
        
        # update the time
        t += dt
    end
    
    return q_traj
end

function naive_leimkuhler_matthews2D_identityD(q0, Vprime, D, div_DDT, tau::Number, m::Integer, dt::Number)
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
    for i in 1:m
        # compute the drift and diffusion coefficients
        grad_V = Vprime(q[1], q[2])
        drift = -grad_V
        Rₖ₊₁ = randn(n)
        diffusion = sqrt(2 * tau) * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[:,i] .= q
        
        # update the time
        t += dt

        # update the noise increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj
end

function naive_leimkuhler_matthews2D(q0, Vprime, D, div_DDT, tau::Number, m::Integer, dt::Number)
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
    for i in 1:m
        # compute the drift and diffusion coefficients
        grad_V = Vprime(q[1], q[2])
        Dq = D(q[1], q[2])
        DDTq = Dq * Dq'
        div_DDTq = div_DDT(q[1], q[2])
        drift = -DDTq * grad_V + tau * div_DDTq 
        Rₖ₊₁ = randn(n)
        diffusion = sqrt(2 * tau) * Dq * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        q += drift * dt + diffusion * sqrt(dt) 
        q_traj[:,i] .= q
        
        # update the time
        t += dt

        # update the noise increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return q_traj
end

function leimkuhler_matthews2D(q0, Vprime, D, div_DDT, tau::Number, m::Integer, dt::Number)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion coefficient at a given configuration
    # tau is the time scale for diffusion
    # m is the number of steps
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    l = length(q0)
    q_traj = zeros(l, m)
    Rₖ = randn(l)

    # number of inner loop steps
    n = 5

    # Get symbolic divergence of diffusion tensor
    div_D = symbolic_matrix_divergence2D(D)

    # tau = 1 in the following
    # simulate
    for i in 1:m
        Dq = D(q)
        grad_V = Vprime(q)
        div_D = div_D(q)
        hat_pₖ₊₁ = Rₖ - sqrt(2 * dt) * Dq * grad_V + sqrt(2 * dt) * div_D
        
        sqrt_h_2 = sqrt(dt / 2)
        inner_step = sqrt_h_2 / n  # Divide by n for each internal RK4 step
        
        # Perform n steps of RK4 integration for hat_qₖ₊₁
        hat_qₖ₊₁ = q # Initialize hat_qₖ₊₁
        for j in 1:n
            # Compute intermediate values
            k1 = inner_step * D(hat_qₖ₊₁) * hat_pₖ₊₁
            k2 = inner_step * D(hat_qₖ₊₁ + 0.5 * k1) * hat_pₖ₊₁
            k3 = inner_step * D(hat_qₖ₊₁ + 0.5 * k2) * hat_pₖ₊₁
            k4 = inner_step * D(hat_qₖ₊₁ + k3) * hat_pₖ₊₁
            
            # Update state using weighted average of intermediate values
            hat_qₖ₊₁ += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        end
        
        # Perform n steps of RK4 integration for qₖ₊₁
        qₖ₊₁ = hat_qₖ₊₁ # Initialize qₖ₊₁
        for j in 1:n
            # Compute intermediate values
            k1 = inner_step * D(qₖ₊₁) * Rₖ₊₁
            k2 = inner_step * D(qₖ₊₁ + 0.5 * k1) * Rₖ₊₁
            k3 = inner_step * D(qₖ₊₁ + 0.5 * k2) * Rₖ₊₁
            k4 = inner_step * D(qₖ₊₁ + k3) * Rₖ₊₁
            
            # Update state using weighted average of intermediate values
            qₖ₊₁ += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        end
        
        # Update the trajectory
        q = qₖ₊₁
        q_traj[:, i] = q
        
        # update the time
        t += dt
    end

    return q_traj
end

function hummer_leimkuhler_matthews2D(q0, Vprime, D, div_DDT, tau::Number, m::Integer, dt::Number)
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
    for i in 1:m
        # compute the drift and diffusion coefficients
        grad_V = Vprime(q[1], q[2])
        Dq = D(q[1], q[2])
        DDTq = Dq * Dq'
        div_DDTq = div_DDT(q[1], q[2])
        drift = -DDTq * grad_V + (3/4) * tau * div_DDTq 
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

function stochastic_heun2D(q0, Vprime, D, div_DDT, tau::Number, m::Integer, dt::Number)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion coefficient at a given configuration
    # tau is the temperature parameter
    # m is the number of steps
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    q_traj = zeros(n, m)
    
    # simulate
    for i in 1:m
        # Compute drift and diffusion coefficients at the current position
        grad_V = Vprime(q[1], q[2])
        Dq = D(q[1], q[2])
        DDTq = Dq * Dq'
        div_DDTq = div_DDT(q[1], q[2])
        drift = -DDTq * grad_V + 0.5 * tau * div_DDTq   #  gradient term gets 0.5 factor to correct for Stratanovich interpretation
        diffusion = sqrt(2 * tau) * Dq
        
        # Compute the predicted next state using Euler-Maruyama
        Rₖ = randn(n)
        q_pred = q + drift * dt + diffusion * Rₖ * sqrt(dt)
        
        # Compute drift and diffusion coefficients at the predicted position
        grad_V_pred = Vprime(q_pred[1], q_pred[2])
        Dq_pred = D(q_pred[1], q_pred[2])
        DDTq_pred = Dq_pred * Dq_pred'
        div_DDTq_pred = div_DDT(q_pred[1], q_pred[2])
        drift_pred = -DDTq_pred * grad_V_pred + 0.5 * tau * div_DDTq_pred   # gradient term gets 0.5 factor to correct for Stratanovich interpretation
        diffusion_pred = sqrt(2 * tau) * Dq_pred
        
        # Compute the corrected next state using a weighted average
        q += 0.5 * (drift + drift_pred) * dt + 0.5 * (diffusion + diffusion_pred) * Rₖ * sqrt(dt)
        q_traj[:,i] .= q
        
        # Update the time
        t += dt
    end
    
    return q_traj
end

function stochastic_heun2D_identityD(q0, Vprime, D, div_DDT, tau::Number, m::Integer, dt::Number)
    # q0 is the initial configuration
    # V is a function that computes the potential energy at a given configuration
    # D is a function that computes the diffusion coefficient at a given configuration
    # tau is the temperature parameter
    # m is the number of steps
    # dt is the time step
    
    # set up
    t = 0.0
    q = copy(q0)
    n = length(q0)
    q_traj = zeros(n, m)
    
    # simulate
    for i in 1:m
        # Compute drift and diffusion coefficients at the current position
        grad_V = Vprime(q[1], q[2])
        drift = -grad_V  
        diffusion = sqrt(2 * tau)
        
        # Compute the predicted next state using Euler-Maruyama
        Rₖ = randn(n)
        q_pred = q + drift * dt + diffusion * Rₖ * sqrt(dt)
        
        # Compute drift and diffusion coefficients at the predicted position
        grad_V_pred = Vprime(q_pred[1], q_pred[2])
        drift_pred = -grad_V_pred 
        
        # Compute the corrected next state using a weighted average
        q += 0.5 * (drift + drift_pred) * dt + diffusion * Rₖ * sqrt(dt)
        q_traj[:,i] .= q
        
        # Update the time
        t += dt
    end
    
    return q_traj
end

end # module Integrators