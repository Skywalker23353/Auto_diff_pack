import jax
import jax.numpy as jnp
import AUTO_DIFF_PACK.chem_source_term_functions as cstf
from scipy.optimize import minimize
from AUTO_DIFF_PACK.logging_util import get_logger

# Get logger instance
logger = get_logger()

# Loss function for regularized least squares
def loss_fn_wrapper(params, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
            A_init, Ea_init, kappa, epsilon, W_k, nu_k, h_f,
            omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg):
    
    A_s, Ea_s = params 
    return loss_fn(A_s, Ea_s, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
            A_init, Ea_init, kappa, epsilon, W_k, nu_k, h_f,
            omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg)
    

def loss_fn(A_s, Ea_s, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
            A_init, Ea_init, kappa, epsilon, W_k, nu_k, h_f,
            omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg): 
    
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

    print("Loss components: MSE_normalized =", mse_normalized, ", Regularization =", reg)
    
    return mse_normalized + reg

def fit_A_and_Ea(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                 A_init, Ea_init, W_k, nu_k, h_f, kappa, epsilon,
                 omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg, init_params):
    
    logger.info("Fitting A and Ea using regularized least squares...")
    logger.debug("Initial A: %.6e, Ea: %.6e, lambda_reg: %.6e", float(A_init), float(Ea_init), lambda_reg)
    
    # Vectorized computation of omega_dot_T for all grid points
    omega_dot_T_vmap = jax.vmap(cstf.omega_dot_T, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (0,0,0,0,0), (0,0,0,0,0), (0,0,0,0,0)))
    
    logger.info("Initial A_opt_s: %.6e, Ea_opt_s: %.6e", float(init_params[0]), float(init_params[1]))
    A_field_temp = A_init * jnp.ones(rhoM.shape, dtype=jnp.float64)
    Ea_field_temp = Ea_init * jnp.ones(rhoM.shape, dtype=jnp.float64)
    print("Omega_dot_T_Model:", omega_dot_T_vmap(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M, 
                                                        A_field_temp, Ea_field_temp, kappa, epsilon, W_k, nu_k, h_f))
    print("Shape of omega_dot_T_Model:", jnp.shape(omega_dot_T_vmap(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M, 
                                                        A_field_temp, Ea_field_temp, kappa, epsilon, W_k, nu_k, h_f)))
    
    # Optimize - regularization term keeps parameters close to initial values
    res = minimize(lambda params: loss_fn_wrapper(params, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                          A_init, Ea_init, kappa, epsilon, W_k, nu_k, h_f,
                                          omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg),
                   init_params,
                   method="nelder-mead",
                #    jac= lambda params: jax.grad(lambda params: loss_fn(params, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                #                           A_init, Ea_init, kappa, epsilon, W_k, nu_k, h_f,
                #                           omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg))(params),
                #    hess=True,
                   options={'disp': True, 'maxiter': 500})
    
    A_s_opt, Ea_s_opt = res.x
    
    logger.debug("Optimization iterations: %d", res.nit)
    logger.debug("Optimization success: %s", res.success)
    logger.info("Optimized A_s: %.6e, Ea_s: %.6e", float(A_s_opt), float(Ea_s_opt))
    logger.info("Final loss: %.6e", float(res.fun)) 

    loss_fn_grad = jax.grad(loss_fn_wrapper())
    res1 = minimize(lambda params: 
                    loss_fn_wrapper(params, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                    A_init, Ea_init, kappa, epsilon, W_k, nu_k, h_f,
                                    omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg),
                    res.x,
                    method="BFGS",
                    jac= lambda params: loss_fn_grad(params, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                    A_init, Ea_init, kappa, epsilon, W_k, nu_k, h_f,
                                    omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg),
                    options={'disp': True, 'maxiter': 500})
    A_s_opt, Ea_s_opt = res1.x
    logger.debug("BFGS Optimization iterations: %d", res1.nit)
    logger.debug("BFGS Optimization success: %s", res1.success)
    logger.info("BFGS Optimized A_s: %.6e, Ea_s: %.6e", float(A_s_opt), float(Ea_s_opt))
    logger.info("BFGS Final loss: %.6e", float(res1.fun))
    
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
