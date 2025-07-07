import numpy as np
import os
import matplotlib.pyplot as plt
from AUTO_DIFF_PACK import read_util as rfu
from AUTO_DIFF_PACK import write_util as wfu
import logging
import sys
# Configure logging to also capture runtime errors
# def log_exceptions(exc_type, exc_value, exc_traceback):
#     if issubclass(exc_type, KeyboardInterrupt):
#         # Allow keyboard interrupts to exit without logging
#         return
#     logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# # Set the custom exception handler
# sys.excepthook = log_exceptions

def plot_error(error, var_idx, k, error_path):
    try:
        x = np.arange(1, (error.shape[2] + 1), 1)
        plt.figure()
        for i in range(error.shape[0]):
            Label = f"$d\\dot{{\\omega}}_{{{i + 1}}}/d\\{k}$"
            logging.info(f"Label: {Label}")
            plt.plot(x, -np.log10(np.abs(error[i][var_idx][:])), label=Label)
        plt.xlabel('n')
        plt.ylabel('-log10(|Error|)')
        title = f"$d\\dot{{\\omega}}_{{K}}/d\\{k}$ computation error"
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(error_path, f"error_wrt_{k}.png"))
        plt.close()
    except Exception as e:
        logging.error(f"An error occurred while plotting error for {k}: {e}", exc_info=True)

def main():

    auto_diff_path = r"docs/Derivs"
    # actual_diff_path = r"docs/Derivs"
    actual_diff_path = r"docs/Derivs"
    error_path = r"docs/Error"
    os.makedirs(error_path, exist_ok=True)

    # Configure logging
    log_file = os.path.join(error_path, "execution_log.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')
    logging.info("Starting the script")
    
    
    #Species
    sp = ["CH4", "O2", "CO2", "H2O"]
    # Variables
    var = ["rho", "T", "Y1", "Y2", "Y3", "Y4"]

    filename_actual = np.zeros((len(sp), len(var)), dtype=object)
    filename_autodiff = np.zeros((len(sp), len(var)), dtype=object)
    for i in range(len(sp)):
        for j in range(len(var)):
            filename_actual[i][j] = f"domega_dot_{i+1}_d{var[j]}_actual.txt"
            filename_autodiff[i][j] = f"domega_dot_{i+1}_d{var[j]}.txt"
    n = 5579

    actual = np.zeros((filename_actual.shape[0],filename_actual.shape[1],n), dtype=np.float64)
    autodiff = np.zeros((filename_autodiff.shape[0],filename_autodiff.shape[1],n), dtype=np.float64)

    #Read the actual and autodiff values

    for i in range(filename_actual.shape[0]):
        for j in range(filename_actual.shape[1]):
            logging.info(f"Reading {filename_actual[i][j]} \n")
            actual[i][j][:] = rfu.read_array_from_file(os.path.join(actual_diff_path, filename_actual[i][j]))
            logging.info(f"Reading {filename_autodiff[i][j]} \n")
            autodiff[i][j][:] = rfu.read_array_from_file(os.path.join(auto_diff_path, filename_autodiff[i][j]))
    
    # Compute the error
    
    error = actual - autodiff
    

    logging.info(f"Error shape: {error.shape}")

    for i in range(error.shape[0]):
        for j in range(error.shape[1]):
           logging.info(f"Maximum error [{i}][{j}][:] {np.max(np.abs(error[i][j][:]))}")
           logging.info(f"Index of maximum error [{i}][{j}][:]: {np.argmax(np.abs(error[i][j][:]))}")

    
    plot_error(error, var_idx=0, k="rho", error_path=error_path)
    # plot_error(error, "T")
    # plot_error(error, "Y1")
    # plot_error(error, "Y2")
    # plot_error(error, "Y3")
    # plot_error(error, "Y4")

    # plt.plot(autodiff[0][:], label='domega_dot_1_drho', color='red')
    # plt.plot(actual[0][:], label='domega_dot_1_drho_actual', color='blue')
    # plt.plot(x, -np.log10(np.abs(error[0][:])), label='domega_dot_1_drho', color='red')
    # plt.plot(x, -np.log10(np.abs(error[1][:])), label='domega_dot_1_dT', color='blue')

    # plt.plot(x, -np.log10(np.abs(error[0][:])), label='CN_rho', color='green')
    # plt.plot(x, -np.log10(np.abs(error[1][:])), label='CN_T', color='magenta')
    # plt.xlim(0, 100)
    

try:
    if __name__ == "__main__":
        main()  # Call the main function
except Exception as e:
    logging.error("An error occurred during execution", exc_info=True)