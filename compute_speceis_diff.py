import jax.numpy as jnp
import jax 
from AUTO_DIFF_PACK import read_util as rfu
from AUTO_DIFF_PACK import chem_source_term_functions as cstf
import os

#Compute derivatives
def main():
    read_path = r"docs/FEHydro_P1"
    #read_path = r"../.FEHydro_P1"
    write_path = r"docs/Derivs"
    #write_path = r"../.FEHydro/Auto_diff"
    os.makedirs(write_path, exist_ok=True) 
    rho_ref = 0.4237
    T_ref = 800#K
    l_ref = 2e-3#m
    U_ref = 65#m/s
    V_ref = l_ref**3
    Cp_ref = 1100e3 #J/kg-K
 
    rhoM = rfu.read_array_from_file(os.path.join(read_path ,'rhobase.txt'))
    rhoM = rho_ref * rhoM
    TM = rfu.read_array_from_file(os.path.join(read_path ,'Tbase.txt'))
    TM = T_ref * TM
    Y0M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase1.txt'))
    Y0M = Y0M
    Y1M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase2.txt'))
    Y1M = Y1M
    Y2M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase3.txt'))
    Y2M = Y2M
    Y3M = rfu.read_array_from_file(os.path.join(read_path ,'Ybase4.txt'))
    Y3M = Y3M
    Y4M = 1 - (Y0M + Y1M + Y2M + Y3M)
    Y = [Y0M,Y1M,Y2M,Y3M]
    species_idx = [0,1,2,3] #CH4, O2, CO2, H2O
    omega_dot_k_scaling = (rho_ref*U_ref)/l_ref
    
    # Compute omega_dot_CH4_differentials---------------------------------------------------
    domega_dot_CH4_drho = jnp.diag(jax.jacfwd(cstf.omega_dot_CH4,0)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_CH4_drho = (domega_dot_CH4_drho*rho_ref)/(omega_dot_k_scaling)
    n = int(len(domega_dot_CH4_drho))   
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_drho.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_drho.txt", 'a') as f:
        for item in domega_dot_CH4_drho:
            f.write("%e\n" % item)
    del domega_dot_CH4_drho

    domega_dot_CH4_dT = jnp.diag(jax.jacfwd(cstf.omega_dot_CH4,1)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_CH4_dT = (domega_dot_CH4_dT*T_ref)/(omega_dot_k_scaling)
    n = int(len(domega_dot_CH4_dT))   
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_dT.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_dT.txt", 'a') as f:
        for item in domega_dot_CH4_dT:
            f.write("%e\n" % item)
    del domega_dot_CH4_dT

    domega_dot_CH4_dY0 = jnp.diag(jax.jacfwd(cstf.omega_dot_CH4,2)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_CH4_dY0 = (domega_dot_CH4_dY0)/(omega_dot_k_scaling)
    n = int(len(domega_dot_CH4_dY0))   
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_dY1.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_dY1.txt", 'a') as f:
        for item in domega_dot_CH4_dY0:
            f.write("%e\n" % item)
    del domega_dot_CH4_dY0

    domega_dot_CH4_dY1 = jnp.diag(jax.jacfwd(cstf.omega_dot_CH4,3)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_CH4_dY1 = (domega_dot_CH4_dY1)/(omega_dot_k_scaling)
    n = int(len(domega_dot_CH4_dY1))   
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_dY2.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_dY2.txt", 'a') as f:
        for item in domega_dot_CH4_dY1:
            f.write("%e\n" % item)
    del domega_dot_CH4_dY1

    domega_dot_CH4_dY2 = jnp.zeros((n), dtype=jnp.float64)
    n = int(len(domega_dot_CH4_dY2))   
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_dY3.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_dY3.txt", 'a') as f:
        for item in domega_dot_CH4_dY2:
            f.write("%e\n" % item)
    del domega_dot_CH4_dY2

    domega_dot_CH4_dY3 = jnp.zeros((n), dtype=jnp.float64)
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_dY4.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_dY4.txt", 'a') as f:
        for item in domega_dot_CH4_dY3:
            f.write("%e\n" % item)
    del domega_dot_CH4_dY3

    domega_dot_CH4_dY4 = jnp.zeros((n), dtype=jnp.float64)
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_dY5.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[0] + 1) + "_dY5.txt", 'a') as f:
        for item in domega_dot_CH4_dY4:
            f.write("%e\n" % item)
    del domega_dot_CH4_dY4

    # Compute omega_dot_O2_differentials---------------------------------------------------
    domega_dot_O2_drho = jnp.diag(jax.jacfwd(cstf.omega_dot_O2,0)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_O2_drho = (domega_dot_O2_drho*rho_ref)/(omega_dot_k_scaling)
    n = int(len(domega_dot_O2_drho))
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_drho.txt", 'w') as f:
        f.write("%d\n" % n) 
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_drho.txt", 'a') as f:
        for item in domega_dot_O2_drho:
            f.write("%e\n" % item)
    del domega_dot_O2_drho

    domega_dot_O2_dT = jnp.diag(jax.jacfwd(cstf.omega_dot_O2,1)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_O2_dT = (domega_dot_O2_dT*T_ref)/(omega_dot_k_scaling)
    n = int(len(domega_dot_O2_dT))
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_dT.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_dT.txt", 'a') as f:
        for item in domega_dot_O2_dT:
            f.write("%e\n" % item)
    del domega_dot_O2_dT

    domega_dot_O2_dY0 = jnp.diag(jax.jacfwd(cstf.omega_dot_O2,2)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_O2_dY0 = (domega_dot_O2_dY0)/(omega_dot_k_scaling)
    n = int(len(domega_dot_O2_dY0))
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_dY1.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_dY1.txt", 'a') as f:
        for item in domega_dot_O2_dY0:
            f.write("%e\n" % item)
    del domega_dot_O2_dY0

    domega_dot_O2_dY1 = jnp.diag(jax.jacfwd(cstf.omega_dot_O2,3)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_O2_dY1 = (domega_dot_O2_dY1)/(omega_dot_k_scaling)
    n = int(len(domega_dot_O2_dY1))
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_dY2.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_dY2.txt", 'a') as f:
        for item in domega_dot_O2_dY1:
            f.write("%e\n" % item)
    del domega_dot_O2_dY1

    domega_dot_O2_dY2 = jnp.zeros((n), dtype=jnp.float64)
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_dY3.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_dY3.txt", 'a') as f:
        for item in domega_dot_O2_dY2:
            f.write("%e\n" % item)
    del domega_dot_O2_dY2

    domega_dot_O2_dY3 = jnp.zeros((n), dtype=jnp.float64)
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_dY4.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_dY4.txt", 'a') as f:
        for item in domega_dot_O2_dY3:
            f.write("%e\n" % item)
    del domega_dot_O2_dY3

    domega_dot_O2_dY4 = jnp.zeros((n), dtype=jnp.float64)
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_dY5.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[1] + 1) + "_dY5.txt", 'a') as f:
        for item in domega_dot_O2_dY4:
            f.write("%e\n" % item)
    del domega_dot_O2_dY4

# Compute omega_dot_CO2_differentials---------------------------------------------------
    domega_dot_CO2_drho = jnp.diag(jax.jacfwd(cstf.omega_dot_CO2,0)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_CO2_drho = (domega_dot_CO2_drho*rho_ref)/(omega_dot_k_scaling)
    n = int(len(domega_dot_CO2_drho))
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_drho.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_drho.txt", 'a') as f:
        for item in domega_dot_CO2_drho:
            f.write("%e\n" % item)
    del domega_dot_CO2_drho

    domega_dot_CO2_dT = jnp.diag(jax.jacfwd(cstf.omega_dot_CO2,1)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_CO2_dT = (domega_dot_CO2_dT*T_ref)/(omega_dot_k_scaling)
    n = int(len(domega_dot_CO2_dT))
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_dT.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_dT.txt", 'a') as f:
        for item in domega_dot_CO2_dT:
            f.write("%e\n" % item)
    del domega_dot_CO2_dT

    domega_dot_CO2_dY0 = jnp.diag(jax.jacfwd(cstf.omega_dot_CO2,2)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_CO2_dY0 = (domega_dot_CO2_dY0)/(omega_dot_k_scaling)
    n = int(len(domega_dot_CO2_dY0))
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_dY1.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_dY1.txt", 'a') as f:
        for item in domega_dot_CO2_dY0:
            f.write("%e\n" % item)
    del domega_dot_CO2_dY0

    domega_dot_CO2_dY1 = jnp.diag(jax.jacfwd(cstf.omega_dot_CO2,3)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_CO2_dY1 = (domega_dot_CO2_dY1)/(omega_dot_k_scaling)
    n = int(len(domega_dot_CO2_dY1))
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_dY2.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_dY2.txt", 'a') as f:
        for item in domega_dot_CO2_dY1:
            f.write("%e\n" % item)
    del domega_dot_CO2_dY1

    domega_dot_CO2_dY2 = jnp.zeros((n), dtype=jnp.float64)
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_dY3.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_dY3.txt", 'a') as f:
        for item in domega_dot_CO2_dY2:
            f.write("%e\n" % item)
    del domega_dot_CO2_dY2

    domega_dot_CO2_dY3 = jnp.zeros((n), dtype=jnp.float64)
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_dY4.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_dY4.txt", 'a') as f:
        for item in domega_dot_CO2_dY3:
            f.write("%e\n" % item)
    del domega_dot_CO2_dY3

    domega_dot_CO2_dY4 = jnp.zeros((n), dtype=jnp.float64)
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_dY5.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[2] + 1) + "_dY5.txt", 'a') as f:
        for item in domega_dot_CO2_dY4:
            f.write("%e\n" % item)
    del domega_dot_CO2_dY4

    # Compute omega_dot_H2O_differentials---------------------------------------------------
    domega_dot_H2O_drho = jnp.diag(jax.jacfwd(cstf.omega_dot_H2O,0)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_H2O_drho = (domega_dot_H2O_drho*rho_ref)/(omega_dot_k_scaling)
    n = int(len(domega_dot_H2O_drho))
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_drho.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_drho.txt", 'a') as f:
        for item in domega_dot_H2O_drho:
            f.write("%e\n" % item)
    del domega_dot_H2O_drho

    domega_dot_H2O_dT = jnp.diag(jax.jacfwd(cstf.omega_dot_H2O,1)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_H2O_dT = (domega_dot_H2O_dT*T_ref)/(omega_dot_k_scaling)
    n = int(len(domega_dot_H2O_dT))
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_dT.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_dT.txt", 'a') as f:
        for item in domega_dot_H2O_dT:
            f.write("%e\n" % item)
    del domega_dot_H2O_dT

    domega_dot_H2O_dY0 = jnp.diag(jax.jacfwd(cstf.omega_dot_H2O,2)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_H2O_dY0 = (domega_dot_H2O_dY0)/(omega_dot_k_scaling)
    n = int(len(domega_dot_H2O_dY0))
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_dY1.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_dY1.txt", 'a') as f:
        for item in domega_dot_H2O_dY0:
            f.write("%e\n" % item)
    del domega_dot_H2O_dY0

    domega_dot_H2O_dY1 = jnp.diag(jax.jacfwd(cstf.omega_dot_H2O,3)(rhoM, TM, Y0M, Y1M, Y2M, Y3M, Y4M))
    domega_dot_H2O_dY1 = (domega_dot_H2O_dY1)/(omega_dot_k_scaling)
    n = int(len(domega_dot_H2O_dY1))
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_dY2.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_dY2.txt", 'a') as f:
        for item in domega_dot_H2O_dY1:
            f.write("%e\n" % item)
    del domega_dot_H2O_dY1

    domega_dot_H2O_dY2 = jnp.zeros((n), dtype=jnp.float64)
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_dY3.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_dY3.txt", 'a') as f:
        for item in domega_dot_H2O_dY2:
            f.write("%e\n" % item)
    del domega_dot_H2O_dY2

    domega_dot_H2O_dY3 = jnp.zeros((n), dtype=jnp.float64)
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_dY4.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_dY4.txt", 'a') as f:
        for item in domega_dot_H2O_dY3:
            f.write("%e\n" % item)
    del domega_dot_H2O_dY3

    domega_dot_H2O_dY4 = jnp.zeros((n), dtype=jnp.float64)
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_dY5.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/domega_dot_" + str(species_idx[3] + 1) + "_dY5.txt", 'a') as f:
        for item in domega_dot_H2O_dY4:
            f.write("%e\n" % item)
    del domega_dot_H2O_dY4

    print("\nEND\n")
if __name__ == "__main__":
        main()
