import jax
import jax.numpy as jnp
import AUTO_DIFF_PACK.chem_source_term_functions as cstf
from jaxopt import ScipyMinimize

# Loss function for regularized least squares
def loss_fn(params, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
            A_init, Ea_init, kappa, epsilon, W_k, nu_k, h_f,
            omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg):
    
    A_s, Ea_s = params 
    
    # Predict omega_dot_T with optimized parameters
    # Expand A and Ea to full field
    A_field = A_s * A_init * jnp.ones(rhoM.shape, dtype=jnp.float64)
    Ea_field = Ea_s * Ea_init * jnp.ones(rhoM.shape, dtype=jnp.float64)
    omega_dot_T_model = omega_dot_T_vmap(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M, 
                                         A_field, Ea_field, kappa, epsilon, W_k, nu_k, h_f)


    # Normalized MSE:
    normalization = (omega_dot_T_LES_rms / jnp.sqrt(N_samples - 1))**2
    mse_normalized = jnp.sum((omega_dot_T_LES - omega_dot_T_model)**2 / normalization)
    
    # Regularization term (penalty on deviation from initial values)
    reg = lambda_reg * ((A_s - 1.0)**2 + (Ea_s - 1.0)**2)
    
    return mse_normalized + reg

def fit_A_and_Ea(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                 A_init, Ea_init, W_k, nu_k, h_f, kappa, epsilon,
                 omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg):
    """
    Fit A and Ea using regularized least squares with vmap.
    
    Args:
        rhoM, TM, Y1M-Y5M: Field data (density, temperature, species mass fractions)
        omega_dot_T_LES: Observed heat release rate from LES
        A_init: Initial pre-exponential factor
        Ea_init: Initial activation energy
        W_k: Molecular weights (tuple of vectors)
        nu_k: Stoichiometric coefficients (tuple of vectors)
        h_f: Enthalpy of formation (tuple of vectors)
        kappa, epsilon: Turbulence quantities
        lambda_reg: Regularization strength
    
    Returns:
        A_opt, Ea_opt: Optimized parameters
    """
    
    # Vectorized computation of omega_dot_T for all grid points
    omega_dot_T_vmap = jax.vmap(cstf.omega_dot_T, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    # Optimize - regularization term keeps parameters close to initial values
    optimizer = ScipyMinimize(fun=lambda params: loss_fn(params, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                                         A_init, Ea_init, kappa, epsilon, W_k, nu_k, h_f,
                                                         omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg),
                              method="L-BFGS-B")
    init_params = jnp.array([1.0, 1.0])  # A_s and Ea_s start at 1.0 (no change from initial)    
    print("Fitting A and Ea using regularized least squares...")
    grad_loss = jax.grad(lambda params: loss_fn(params, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                    A_init, Ea_init, kappa, epsilon, W_k, nu_k, h_f,
                                    omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg))
    
    result = optimizer.run(init_params)
    
    A_s_opt, Ea_s_opt = result.params
    
    # Compute gradient at optimal point
    grad_at_opt = grad_loss(result.params)
    optimal_loss = float(result.state.fun_val)
    
    # Normalize gradients by optimal loss value
    grad_normalized = grad_at_opt / (optimal_loss + 1e-10)  # Add small epsilon to avoid division by zero
    
    A_opt = float(A_s_opt * A_init)
    Ea_opt = float(Ea_s_opt * Ea_init)
    
    print(f"Optimization Results:")
    print(f"  Initial A: {float(A_init):.6e}, Optimized A: {A_opt:.6e}")
    print(f"  Initial Ea: {float(Ea_init):.6e}, Optimized Ea: {Ea_opt:.6e}")
    print(f"  Optimal loss: {optimal_loss:.6e}")
    print(f"  Gradient at optimum: dL/dA_s = {float(grad_at_opt[0]):.6e}, dL/dEa_s = {float(grad_at_opt[1]):.6e}")
    print(f"  Normalized gradient (grad/loss): dL/dA_s = {float(grad_normalized[0]):.6e}, dL/dEa_s = {float(grad_normalized[1]):.6e}")
    print(f"  Gradient magnitude: {float(jnp.linalg.norm(grad_at_opt)):.6e}")
    print(f"  Gradient norm / loss: {float(jnp.linalg.norm(grad_at_opt) / (optimal_loss + 1e-10)):.6e}\n")
    
    return A_s_opt, Ea_s_opt

def compute_rmse(omega_dot_T_LES, omega_dot_T_model):
    """Compute RMSE"""
    rmse = jnp.sqrt(jnp.mean((omega_dot_T_LES - omega_dot_T_model)**2))
    return float(rmse)

def compute_nrmse(omega_dot_T_LES, omega_dot_T_model):
    """Compute Normalized RMSE (relative to observed data range)"""
    rmse = jnp.sqrt(jnp.mean((omega_dot_T_LES - omega_dot_T_model)**2))
    data_range = jnp.max(omega_dot_T_LES)
    nrmse = rmse / data_range
    return float(nrmse)

