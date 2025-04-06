import numpy as np
import os
import matplotlib.pyplot as plt
from AUTO_DIFF_PACK import read_util as rfu

def main():
    auto_diff_path = r"docs/Derivs"
    actual_diff_path = r"docs/Derivs"

    filename_actual = ['omega_dot_Y1_1_actual.txt', 'omega_dot_Y0_0_actual.txt', 'omega_dot_Y0_0_actual.txt', 'omega_dot_Y0_0_actual.txt']
    filename_autodiff = ['omega_dot_Y_1_1.txt', 'omega_dot_T_1.txt', 'omega_dot_T_2.txt', 'omega_dot_T_3.txt']
    n = 5579

    actual = np.zeros((len(filename_actual), n), dtype=np.float64)
    autodiff = np.zeros((len(filename_autodiff), n), dtype=np.float64)

    for i in range(0, len(filename_actual)):
        actual[i,:] = rfu.read_array_from_file(os.path.join(actual_diff_path, filename_actual[i]))
        autodiff[i,:] = rfu.read_array_from_file(os.path.join(auto_diff_path, filename_autodiff[i]))

    # Compute the error
    error = np.zeros(((actual.shape[0]), (actual.shape[1])), dtype=np.float64)
    for i in range(0,(actual.shape[0]-1)):
        for j in range(0,actual.shape[1]-1):
            error[i][j] = actual[i][j] - autodiff[i][j]
    error = error + 1e-16
    x = np.arange(1,error.shape[1]+1)
    plt.figure()
    plt.plot(x, -np.log10(np.abs(error[0][:])), label='omega_dot_Y1_1', color='red')
    # plt.plot(x, -np.log10(np.abs(error[1][:])), label='omega_dot_T_1', color='blue')
    # plt.plot(x, -np.log10(np.abs(error[2][:])), label='omega_dot_T_2', color='green')
    # plt.plot(x, -np.log10(np.abs(error[3][:])), label='omega_dot_T_3', color='magenta')
    plt.xlabel('X')
    plt.ylabel('-log10(|Error|)')
    plt.title('C computation error')
    # plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()  # Call the main function