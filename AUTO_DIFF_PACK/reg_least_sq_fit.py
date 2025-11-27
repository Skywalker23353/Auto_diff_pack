import jax
import jax.numpy as jnp
import AUTO_DIFF_PACK.chem_source_term_functions as cstf
from scipy.optimize import minimize
from AUTO_DIFF_PACK.logging_util import get_logger

# Get logger instance
logger = get_logger()

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
    
    logger.info("Fitting A and Ea using regularized least squares...")
    logger.debug("Initial A: %.6e, Ea: %.6e, lambda_reg: %.6e", A_init, Ea_init, lambda_reg)
    
    # Vectorized computation of omega_dot_T for all grid points
    omega_dot_T_vmap = jax.vmap(cstf.omega_dot_T, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    init_params = jnp.array([1.0, 1.0])
    logger.info("Initial A_opt_s: %.6e, Ea_opt_s: %.6e", float(init_params[0]), float(init_params[1]))
    
    # Optimize - regularization term keeps parameters close to initial values
    res = minimize(lambda params: loss_fn(params, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                          A_init, Ea_init, kappa, epsilon, W_k, nu_k, h_f,
                                          omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg),
                   init_params,
                   method="L-BFGS-B")
    
    A_s_opt, Ea_s_opt = res.x
    
    logger.debug("Optimization iterations: %d", res.nit)
    logger.debug("Optimization success: %s", res.success)
    logger.info("Optimized A_s: %.6e, Ea_s: %.6e", float(A_s_opt), float(Ea_s_opt))
    logger.info("Final loss: %.6e", float(res.fun))
    
    return A_s_opt, Ea_s_opt

def compute_rmse(omega_dot_T_LES, omega_dot_T_model):
    """Compute RMSE"""
    rmse = jnp.sqrt(jnp.mean((omega_dot_T_LES - omega_dot_T_model)**2))
    rmse_float = float(rmse)
    logger.info("RMSE: %.6e", rmse_float)
    return rmse_float

def compute_nrmse(omega_dot_T_LES, omega_dot_T_model):
    """Compute Normalized RMSE (relative to observed data range)"""
    rmse = jnp.sqrt(jnp.mean((omega_dot_T_LES - omega_dot_T_model)**2))
    data_max = jnp.max(omega_dot_T_LES)
    nrmse = rmse / data_max
    nrmse_float = float(nrmse)
    logger.info("Normalized RMSE: %.6f", nrmse_float)
    logger.debug("RMSE numerator: %.6e, data_max: %.6e", float(rmse), float(data_max))
    return nrmse_float

