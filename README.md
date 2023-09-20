# Transforms-For-Brownian-Dynamics

This repository is a Julia implementation of Brownian dynamics integrators with coordinate transforms to constant diffusion.

This codebase was developed as part of the article "Numerical Methods with Coordinate Transforms for Efficient Brownian Dynamics Simulations", which can be found on the ArXiv [here](https://arxiv.org/abs/2307.02913).

**Motivation**
Many numerical integrators of Stochastic Differential Equations (SDEs) work well for processes that have a constant noise (a.k.a. additive noise) but significantly lose performance, or even fail to converge for processes with variable noise (a.k.a. multiplicative noise). One solution, if and when it is possible, is to transform the multiplicative noise process into an additive noise process through a reversible transform of the space or time coordinates. The additive noise process can then be more efficiently sampled with a more traditional SDE integrator and the resulting trajectories can be reweighted in order to reconstruct the sampling or dynamical properties of the original process of interest. 

Brownian dynamics is one of the most important classes of SDE process, with applications across the physical, biological, and data-driven sciences. This codebase provides a toolbox for running high-efficiency simulations of Brownian dynamics SDE processes with implementations of the numerical integrators, coordinate transforms, and reweighting formulas.

