# Transforms-For-Brownian-Dynamics

*Brownian dynamics integrators with coordinate transforms to constant diffusion, implemented in Julia.*

This codebase was developed for the paper "Numerical Methods with Coordinate Transforms for Efficient Brownian Dynamics Simulations", which can be found [here](https://arxiv.org/abs/2307.02913).

## Motivation
Numerical integrators of Stochastic Differential Equations (SDEs) work well for constant (additive) noise but often lose performance, or fail to converge, for variable (multiplicative) noise. One solution, if and when it is possible, is to apply reversible transforms in space or time to convert a variable noise process into a constant noise process, which can be more efficiently sampled with traditional SDE integrators. Then, trajectories can be reweighted and the statistical and dynamical properties of the original process can be reconstructed. 

**Brownian dynamics** is one of the most important classes of SDE process, with applications across the physical, biological, and data-driven sciences. This codebase provides a Julia toolbox for running high-efficiency simulations of Brownian dynamics processes with coordinate transforms.

## Installation

To install this project, follow these steps:

1. Clone the repository.

`git clone https://github.com/dominicp6/Transforms-For-Brownian-Dynamics`

2. Navigate to the project directory.

3. Install the required Julia packages:

[list of packages]

Done!

## Usage



## Background Theory
What follows is an introduction. See the [paper](https://arxiv.org/abs/2307.02913) for all technical details.

### Brownian Dynamics
Brownian dynamics is defined through an Ito stochastic differential equation (SDE), which in one dimension reads

$$
    dx_t = - D(x_t) \frac{dV(x_t)}{dx}dt + kT \frac{d D(x_t)}{dx}dt + \sqrt{2 kT D(x_t)} dW_t,
$$

where $`t \in \mathbb{R}_{>0}`$ is time, $`x_t \in \mathbb{R}`$ is the state variable, $`W_t`$ is a one-dimensional Wiener process, $`V : \mathbb{R} \xrightarrow{} \mathbb{R}`$ is a potential energy function, $`D : \mathbb{R} \xrightarrow{} \mathbb{R}_{>0}`$ is the diffusion coefficient, $`k`$ is the Boltzmann constant and $`T`$ is the temperature in degrees Kelvin. Note that the diffusion coefficient $`D(x)`$ is a function of $x$ which means that we have configuration-dependent noise, also known as multiplicative noise. 

In higher dimensions, 

$$
   d\mathbf{X}_t = -(\mathbf{D}(\mathbf{X}_t)\mathbf{D}(\mathbf{X}_t)^T) \nabla V(\mathbf{X}_t) dt + kT \text{div}(\mathbf{D}\mathbf{D}^T)(\mathbf{X}_t) dt + \sqrt{2 kT} \mathbf{D}(\mathbf{X}_t)d\mathbf{W}_t,
$$

where $`\mathbf{X}_t \in \mathbb{R}^n`$ is the state variable, $`\mathbf{W}_t`$ is an n-dimensional Wiener process, $`V: \mathbb{R}^n \xrightarrow{} \mathbb{R}`$ is a potential function, and $`\mathbf{D}\mathbf{D}^T: \mathbb{R}^n \xrightarrow{} \mathbb{R}^n \times \mathbb{R}^n`$ is a configuration-dependent diffusion tensor that is everywhere positive definite.

### Transforms
#### Lamperti Transform (Global Coordinate Transform)

Consider multivariate Brownian dynamics with $`\mathbf{D}`$ matrix

$$
  \mathbf{D}(\mathbf{X})_{ij} = D_i  (X_i) R _{ij},
$$

where $`R_{ij}`$ in an invertible, constant matrix. For this class of diffusion,  a Lamperti transform to unit diffusion can be constructed, however, the transformed dynamics is only Brownian dynamics with a conservative drift force if $`\mathbf{R}`$ is proportional to the identity. Specifically, when $`\mathbf{D}(\mathbf{X})_{ij} = D_i(X_i)\delta_{ij}`$, the Lamperti-transformed process is $`Y_{i,t} = \sqrt{2kT} \int_{x_0}^{X_{i,t}} \frac{1}{D_i(x)} dx := \sqrt{2kT} \phi_i(X_{i,t})`$, and obeys

$$
  dY_{i,t} = -\nabla_{Y_i} \hat{V}({\mathbf{Y}})dt + \sqrt{2kT}dW_i,
$$

with an effective potential 

$$
    \hat{V}(\mathbf{Y}) = V(\phi^{-1}(\mathbf{Y})) - kT \sum_{k=1}^n \ln D_k (\phi^{-1}_k (Y _{k,t})).
$$

Phase-space averages with respect to the invariant measure, $`\rho(\mathbf{X})`$, of the original process can be recovered through

$$
\int_{\mathbb{R}^n} f(\mathbf{X})\rho(\mathbf{X}) d\mathbf{X} = \lim_{T \rightarrow \infty} \frac{1}{T} \int_{t=0}^T f(\phi^{-1}(\mathbf{Y}_t)) dt.
$$

In the above, the map $`\phi^{-1}: \mathbb{R}^n \rightarrow \mathbb{R}`$ is constructed by individually applying $`\phi_i^{-1}`$ to each component of its argument, $`1 \leq i \leq n`$.

#### Time-Rescaling Transform

Consider multivariate Brownian dynamics with $\mathbf{D}$ matrix

$$
\mathbf{D}(\mathbf{X}) = D(\mathbf{X})\mathbf{R},
$$

where $`\mathbf{R}`$ is an invertible matrix. For this class of variable diffusion, a time-rescaling to Brownian dynamics unit diffusion can be constructed. The time-rescaled process is given by $`\mathbf{Y}_\tau = \mathbf{R}^{-1}\mathbf{X}_\tau`$  where $`\frac{dt}{d\tau} = g(\mathbf{X}) := 1/D^2(\mathbf{X})`$ and it obeys

$$
d\mathbf{Y}_{\tau} = - \nabla _{\mathbf{Y}}\hat{V}(\mathbf{Y})dt + \sqrt{2kT}d\mathbf{W},
$$

with an effective potential

$$
\hat{V}(\mathbf{Y}) = V(\mathbf{RY})- 2kT \ln D(\mathbf{RY}).
$$

Phase-space averages with respect to the invariant measure, $`\rho(\mathbf{X})`$, of the original process can be recovered through

$$
\int_{\mathbb{R}^n} f(\mathbf{X}) \rho(\mathbf{X}) d\mathbf{X} = \lim_{T \rightarrow \infty} \frac{\int_{\tau=0}^{T} f(\mathbf{R}\mathbf{Y}_ \tau) g(\mathbf{R}\mathbf{Y}_ \tau) d\tau}{\int_{\tau=0}^{T} g(\mathbf{R}\mathbf{Y}_\tau) d\tau}.
$$

*Note: The Lamperti and time-rescaling transforms can also be combined, see the paper for details.*

## Using the Code



### Code Structure


