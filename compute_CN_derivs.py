# This script computes the derivatives of the volumetric heat release rate with respect to the state variables
from AUTO_DIFF_PACK import f_w_functions as fw
from AUTO_DIFF_PACK import read_util as rfu
import os
import jax.numpy as jnp

#Compute derivatives
def main():
    filename_A_Arrhenius = r"docs/pre-exponential_A.txt"
    with open(filename_A_Arrhenius, 'r') as file:
        A = jnp.array([float(file.readline().strip())], dtype=jnp.float64)

    filename_Qbar = r"docs/Mean_Qbar.txt"
    with open(filename_Qbar, 'r') as file:
        Q_bar = jnp.array([float(file.readline().strip())], dtype=jnp.float64)

    read_path = r"docs/FEHydro_P1"
    # read_path = r"../.FEHydro_P1"
    write_path = r"docs/Derivs"
    # write_path = r"../.FEHydro/Baseflow_CN_P1"
    os.makedirs(write_path, exist_ok=True) 
    rho_ref = 0.4237
    T_ref = 800#K
    l_ref = 2e-3#m
    U_ref = 65#m/s
    V_ref = l_ref**3
 
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
    #Y5M = read_array_from_file(read_path + '/Ybase5.txt')
    #Y5M = Y5M

    Y = [Y0M,Y1M,Y2M,Y3M]
    species_idx = [0,1,2,3] #CH4, O2, CO2, H2O
    q = [2,3,4,5,6,7] # q[0] = rho, q[1] = T, q[2] = Y0, q[3] = Y1, q[4] = Y2, q[5] = Y3
    scalefactor = 1
    eps = 0 
    omega_dot_k_scaling = (rho_ref*U_ref)/l_ref
    # Compute C_rho---------------------------------------------------
    C_rho = fw.return_wT_deriv(A,q[0],rhoM, TM, Y0M, Y1M, Y2M, Y3M)
    C_rho = C_rho*rho_ref*V_ref/Q_bar
    n = int(len(C_rho))
    with open(write_path + "/CN_rho.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/CN_rho.txt", 'a') as f:
        for item in C_rho:
            f.write("%e\n" % item)
    del C_rho
    # # %Compute C_T---------------------------------------------------
    C_T = fw.return_wT_deriv(A,q[1], rhoM, TM, Y0M, Y1M, Y2M, Y3M)
    C_T = C_T*T_ref*V_ref/Q_bar
    with open(write_path + "/CN_T.txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/CN_T.txt", 'a') as f:
        for item in C_T:
            f.write("%e\n" % item)
    del C_T
    # %Compute C_Y---------------------------------------------------
    for k in range(2, (len(Y)+2)):
            C_Y = fw.return_wT_deriv(A, q[k], rhoM, TM, Y0M, Y1M, Y2M, Y3M)
            C_Y = C_Y*V_ref/Q_bar
            # Increase the size of C_sq_Y by 1 and insert the length as the first element
            with open(write_path + "/CN_Y" + str(k-1) + ".txt", 'w') as f:
                f.write("%d\n" % n)
            with open(write_path + "/CN_Y" + str(k-1) + ".txt", 'a') as f:
                for item in C_Y:
                    f.write("%e\n" % item)
            del C_Y

    # Compute omega_dot_rho---------------------------------------------------
    for i in range(len(species_idx)):
        omega_dot_rho = fw.return_omega_dot_k(A,species_idx[i],q[0],rhoM, TM, Y0M, Y1M, Y2M, Y3M)
        omega_dot_rho = (omega_dot_rho*rho_ref)/(omega_dot_k_scaling)
        n = int(len(omega_dot_rho))
        with open(write_path + "/omega_dot_" + str(species_idx[i] + 1) + "_rho.txt", 'w') as f:
            f.write("%d\n" % n)
        with open(write_path + "/omega_dot_" + str(species_idx[i] + 1) + "_rho.txt", 'a') as f:
            for item in omega_dot_rho:
                f.write("%e\n" % item)
        del omega_dot_rho
    # %Compute omega_dot_T---------------------------------------------------
    for i in range(len(species_idx)):
        omega_dot_T = fw.return_omega_dot_k(A,species_idx[i],q[1],rhoM, TM, Y0M, Y1M, Y2M, Y3M)
        omega_dot_T = (omega_dot_T*T_ref)/(omega_dot_k_scaling)
        n = int(len(omega_dot_T))
        with open(write_path + "/omega_dot_" + str(species_idx[i] + 1) + "_T.txt", 'w') as f:
            f.write("%d\n" % n)
        with open(write_path + "/omega_dot_" + str(species_idx[i] + 1) + "_T.txt", 'a') as f:
            for item in omega_dot_T:
                f.write("%e\n" % item)
        del omega_dot_T
    # %Compute omega_dot_Y---------------------------------------------------
    for i in range(len(species_idx)):
        for j in range(2, (len(Y)+2)):
            omega_dot_Y = fw.return_omega_dot_k(A,species_idx[i],q[j],rhoM, TM, Y0M, Y1M, Y2M, Y3M)
            omega_dot_Y = (omega_dot_Y)/(omega_dot_k_scaling)
            n = int(len(omega_dot_Y))
            with open(write_path + "/omega_dot_" + str(species_idx[i] + 1) + "_Y" + str(j-1) + ".txt", 'w') as f:
                f.write("%d\n" % n)
            with open(write_path + "/omega_dot_" + str(species_idx[i] + 1) + "_Y" + str(j-1) + ".txt", 'a') as f:
                for item in omega_dot_Y:
                    f.write("%e\n" % item)
            del omega_dot_Y
    
    print("\nEND\n")
if __name__ == "__main__":
        main()
