import numpy as np
import os
import matplotlib.pyplot as plt
from AUTO_DIFF_PACK import read_util as rfu
from AUTO_DIFF_PACK import write_util as wfu

def main():
    auto_diff_path = r"docs/Derivs"
    # actual_diff_path = r"docs/Derivs"
    actual_diff_path = r"docs/Baseflow_CN_P1"
    error_path = r"docs/Error"

    os.makedirs(error_path, exist_ok=True)

    # filename_actual = ["domega_dot_1_drho_actual.txt", "domega_dot_1_dT_actual.txt"]#, "domega_dot_1_dY1_actual.txt", "domega_dot_1_dY2_actual.txt"]
    # filename_autodiff = ["domega_dot_1_drho.txt", "domega_dot_1_dT.txt"]#, "domega_dot_1_dY1.txt", "domega_dot_1_dY2.txt"]
    filename_actual = ["CN_rho.txt", "CN_T.txt"]
    filename_autodiff = filename_actual
    n = 5579

    actual = np.zeros((len(filename_actual), n), dtype=np.float64)
    autodiff = np.zeros((len(filename_autodiff), n), dtype=np.float64)

    for i in range(0, len(filename_actual)):
        actual[i,:] = rfu.read_array_from_file(os.path.join(actual_diff_path, filename_actual[i]))
        autodiff[i,:] = rfu.read_array_from_file(os.path.join(auto_diff_path, filename_autodiff[i]))

    # Compute the error
    # error = np.zeros(((actual.shape[0]), (actual.shape[1])), dtype=np.float64)
    error = actual - autodiff
    print("Maximum error [0][:]: ", np.max(np.abs(error[0][:])))
    print("Index of maximum error [0][:]: ", np.argmax(np.abs(error[0][:])))
    print("Maximum error [1][:]: ", np.max(np.abs(error[1][:])))
    print("Index of maximum error [1][:]: ", np.argmax(np.abs(error[1][:])))
    error = error #+ 1e-16
    # wfu.write_to_file(error_path, "domega_dot_1_drho_error.txt", error[0][:])
    # wfu.write_to_file(error_path, "domega_dot_1_dT_error.txt", error[1][:])
    x = np.arange(1,(error.shape[1]+1),1)
    plt.figure()
    # plt.plot(autodiff[0][:], label='domega_dot_1_drho', color='red')
    # plt.plot(actual[0][:], label='domega_dot_1_drho_actual', color='blue')
    # plt.plot(x, -np.log10(np.abs(error[0][:])), label='domega_dot_1_drho', color='red')
    # plt.plot(x, -np.log10(np.abs(error[1][:])), label='domega_dot_1_dT', color='blue')

    plt.plot(x, -np.log10(np.abs(error[0][:])), label='CN_rho', color='green')
    plt.plot(x, -np.log10(np.abs(error[1][:])), label='CN_T', color='magenta')
    # plt.xlim(0, 100)
    plt.xlabel('X')
    # plt.ylabel('Differential Value')
    plt.ylabel('-log10(|Error|)')
    plt.title('$\dot{\omega}^{''}_{k}$ computation error')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()  # Call the main function