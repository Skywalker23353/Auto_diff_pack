import jax
import jax.numpy as jnp
import AUTO_DIFF_PACK.chem_source_term_functions as cstf
from scipy.optimize import minimize
from AUTO_DIFF_PACK.logging_util import get_logger

# Get logger instance
logger = get_logger()

# Loss function for regularized least squares
def loss_fn_wrapper(params, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
            A_val, Ea_val, kappa, epsilon, W_k, nu_k, h_f,
            omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg, T_c_st, delta_st):
    
    A_s, Ea_s, T_c, delta, model_uncty = params 
    return loss_fn(A_s, Ea_s, T_c, delta, model_uncty, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
            A_val, Ea_val, kappa, epsilon, W_k, nu_k, h_f,
            omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg, T_c_st, delta_st)
    

def loss_fn(A_s, Ea_s, T_c, delta, model_uncty, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
            A_val, Ea_val, kappa, epsilon, W_k, nu_k, h_f,
            omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg, T_c_st, delta_st): 
    
    # Expand A and Ea to full field
    A_field = A_s * A_val * jnp.ones(rhoM.shape, dtype=jnp.float64)
    Ea_field = Ea_s * Ea_val * jnp.ones(rhoM.shape, dtype=jnp.float64)
    model_uncty_field = model_uncty * jnp.ones(rhoM.shape, dtype=jnp.float64)
    omega_dot_T_model = omega_dot_T_vmap(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M, 
                                         A_field, Ea_field, kappa, epsilon, W_k, nu_k, h_f)


    # Normalized MSE:
    normalization = (omega_dot_T_LES_rms / jnp.sqrt(N_samples - 1))**2 + model_uncty_field**2
    print("SIze of normalization:", jnp.shape(normalization))
    mse_normalized = (omega_dot_T_LES - omega_dot_T_model)**2 / normalization
    J_l = jnp.mean(mse_normalized)

    uncty_norm = 0.5 * jnp.log(2 * jnp.pi) + 0.5 * jnp.sum(jnp.log(normalization))
    J_l += uncty_norm

    # Regularization term (penalty on deviation from initial values)
    reg = lambda_reg * ((A_s - 1.0)**2 + (Ea_s - 1.0)**2 + (T_c - T_c_st)**2 + (delta - delta_st)**2)

    print("Loss components: MSE_normalized =", J_l, ", Regularization =", reg)
    
    return J_l + reg

def fit_A_and_Ea(rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                 A_val, Ea_val, W_k, nu_k, h_f, kappa, epsilon,
                 omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg, init_params, T_c_st, delta_st):
    
    logger.info("Fitting A and Ea using regularized least squares...")
    logger.debug("Initial A: %.6e, Ea: %.6e, lambda_reg: %.6e", float(A_val), float(Ea_val), lambda_reg)
    
    # Vectorized computation of omega_dot_T for all grid points
    omega_dot_T_vmap = jax.vmap(cstf.omega_dot_T, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (0,0,0,0,0), (0,0,0,0,0), (0,0,0,0,0)))
    
    logger.info("Initial A_opt_s: %.6e, Ea_opt_s: %.6e", float(init_params[0]), float(init_params[1]))

    # Optimize - regularization term keeps parameters close to initial values
    res = minimize(lambda params: loss_fn_wrapper(params, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                          A_val, Ea_val, kappa, epsilon, W_k, nu_k, h_f,
                                          omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg,T_c_st, delta_st),
                   init_params,
                   method="nelder-mead",
                   options={'disp': True})
    
    A_s_opt, Ea_s_opt, model_uncty_opt, T_c, delta = res.x
    
    logger.debug("Optimization iterations: %d", res.nit)
    logger.debug("Optimization success: %s", res.success)
    logger.info("Optimized A_s: %.6e, Ea_s: %.6e, model_uncty: %.6e", float(A_s_opt), float(Ea_s_opt), float(model_uncty_opt))
    logger.info("Optimized T_c: %.6e, delta: %.6e", float(T_c), float(delta))
    logger.info("Final loss: %.6e", float(res.fun)) 

    loss_fn_grad = jax.grad(loss_fn_wrapper)
    res1 = minimize(lambda params: 
                    loss_fn_wrapper(params, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                    A_val, Ea_val, kappa, epsilon, W_k, nu_k, h_f,
                                    omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg, T_c_st, delta_st),
                    res.x,
                    method="BFGS",
                    jac= lambda params: loss_fn_grad(params, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                    A_val, Ea_val, kappa, epsilon, W_k, nu_k, h_f,
                                    omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg, T_c_st, delta_st),
                    options={'disp': True, 'gtol': 1e-6, 'ftol': 1e-6})
    
    hessian = jax.hessian(loss_fn_wrapper)
    hess_evaluated = hessian(res1.x, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                    A_val, Ea_val, kappa, epsilon, W_k, nu_k, h_f,
                                    omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg, T_c_st, delta_st)
    logger.debug("Hessian at optimum:\n%s", hess_evaluated) 
    # Symmetrize the Hessian and check positive-definiteness
    sym_hess = 0.5 * (hess_evaluated + hess_evaluated.T)

    # Compute eigenvalues (real for a symmetric matrix)
    eigvals = jnp.linalg.eigvalsh(sym_hess)
    eigvals_host = jax.device_get(eigvals)
    min_eig = float(jax.device_get(jnp.min(eigvals)))

    is_pos_def = bool(min_eig > 1e-12)

    logger.info("Hessian eigenvalues: %s", eigvals_host.tolist())
    logger.info("Hessian symmetric min eigenvalue: %.6e", min_eig)
    logger.info("Hessian positive definite (min_eig > 1e-12): %s", is_pos_def)

    # Robust check via Cholesky
    try:
        _ = jnp.linalg.cholesky(sym_hess)
        logger.info("Cholesky decomposition succeeded -> Hessian is positive definite.")
    except Exception as e:
        logger.info("Cholesky decomposition failed -> Hessian is NOT positive definite. Exception: %s", str(e))
    
    gradient = loss_fn_grad(res1.x, omega_dot_T_vmap, rhoM, TM, Y1M, Y2M, Y3M, Y4M, Y5M,
                                    A_val, Ea_val, kappa, epsilon, W_k, nu_k, h_f,
                                    omega_dot_T_LES, omega_dot_T_LES_rms, N_samples, lambda_reg, T_c_st, delta_st)
    gradient_norm = jnp.linalg.norm(gradient)
    logger.info("Gradient at optimum: %s", jax.device_get(gradient).tolist())
    logger.info("Gradient norm at optimum: %.6e", float(gradient_norm))

    diag_Hess = jnp.diag(hess_evaluated)
    logger.info("Hessian diagonal at optimum: %s", jax.device_get(diag_Hess).tolist())
    hessian_det = (diag_Hess[0] * diag_Hess[1]) ** (-1/2)
    logger.info("Volume of the 1 sigma ellipsoid inverse square root: %.6e", float(hessian_det))
    
    A_s_opt, Ea_s_opt, model_uncty_opt, T_c, delta = res1.x
    logger.debug("BFGS Optimization iterations: %d", res1.nit)
    logger.debug("BFGS Optimization success: %s", res1.success)
    logger.info("BFGS Optimized A_s: %.6e, Ea_s: %.6e, model_uncty: %.6e", float(A_s_opt), float(Ea_s_opt), float(model_uncty_opt))
    logger.info("BFGS Final loss: %.6e", float(res1.fun))
    
    return A_s_opt, Ea_s_opt, T_c, delta

# def compute_rmse(omega_dot_T_LES, omega_dot_T_model):
#     """Compute RMSE"""
#     rmse = jnp.sqrt(jnp.mean((omega_dot_T_LES - omega_dot_T_model)**2))
#     rmse_float = float(rmse)
#     logger.info("RMSE: %.6e", rmse_float)
#     return rmse_float

# def compute_nrmse(omega_dot_T_LES, omega_dot_T_model):
#     """Compute Normalized RMSE (relative to observed data range)"""
#     rmse = jnp.sqrt(jnp.mean((omega_dot_T_LES - omega_dot_T_model)**2))
#     data_max = jnp.max(omega_dot_T_LES)
#     nrmse = rmse / data_max
#     nrmse_float = float(nrmse)
#     logger.info("Normalized RMSE: %.6f", nrmse_float)
#     logger.debug("RMSE numerator: %.6e, data_max: %.6e", float(rmse), float(data_max))
#     return nrmse_float
