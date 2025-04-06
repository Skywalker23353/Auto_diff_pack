# This package created the derivatives of Heat release rate and species production rate at mean flow values using Autodiff.
# The jax function used to compute the derivatives is jacfwd(). It is for a vector valued function which returns a vector although jax.grad() should have been used since it is meant for scalar valued function which is relevant for our case.
# This package has function from the existing CN_derivatives code that I had written to compute derivatives for heat release rate norm.
