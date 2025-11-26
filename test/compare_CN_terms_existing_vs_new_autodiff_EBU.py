import numpy as np
import os
import matplotlib.pyplot as plt
from AUTO_DIFF_PACK import read_util as rfu
from AUTO_DIFF_PACK import write_util as wfu
import logging

def plot_error(data,var_name, save_path,save_plot):
    """
    Plot the error for each variable and save the plot.
    
    Parameters:
    error (np.ndarray): The error array.
    var (list): List of variable names.
    error_path (str): Path to save the error plots.
    """
    x = np.arange(1, (data.shape[0] + 1), 1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, -np.log10(np.abs(data)), label=var_name)
    
    plt.xlabel('n')
    plt.ylabel('-log10(|Error|)')
    title = "CN fields Computation error"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Save the plot
    if save_plot:
        plt.savefig(os.path.join(save_path, "error_plot.png"))
        plt.close()

def plot_data(data,var_name, save_path,save_plot):
    """
    Plot data for each variable and save the plot.
    
    Parameters:
    data (np.ndarray): The error array.
    var (list): List of variable names.
    save_path (str): Path to save the plots.
    """
    x = np.arange(1, (data.shape[0] + 1), 1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, -np.log10(np.abs(data)), label=var_name)
    
    plt.xlabel('n')
    title = "CN fields"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Save the plot
    if save_plot:
        plt.savefig(os.path.join(save_path, "data_plot" + var_name + ".png"))
        plt.close()

def main():

    auto_diff_path = r"docs/Derivs_EBU"
    actual_diff_path = r"docs/Derivs_EBU"
    # actual_diff_path = r"docs/Baseflow_CN_P1_latest"
    error_path = r"docs/Error"
    os.makedirs(error_path, exist_ok=True)
    
    # actual = rfu.read_array_from_file(os.path.join(actual_diff_path, "CN_rho.txt"))
    # autodiff = rfu.read_array_from_file(os.path.join(auto_diff_path, "CN_rho.txt"))
    # TERMS
    var_autodiff = ["domega_dot_1_drho"]#, "domega_dot_1_dY2","domega_dot_2_drho","domega_dot_2_dY2","domega_dot_3_drho","domega_dot_3_dY2","domega_dot_4_drho","domega_dot_4_dY2"]
    
    n = 5579
    filename_actual = np.zeros((len(var_autodiff)), dtype=object)
    filename_autodiff = np.zeros((len(var_autodiff)), dtype=object)
    for i in range(len(var_autodiff)):
        filename_actual[i] = f"{var_autodiff[i]}_actual.txt"
        filename_autodiff[i] = f"{var_autodiff[i]}.txt"
    

    actual = np.zeros((filename_actual.shape[0],n), dtype=np.float64)
    autodiff = np.zeros((filename_autodiff.shape[0],n), dtype=np.float64)

    #Read the actual and autodiff values

    for i in range(filename_actual.shape[0]):
        logging.info(f"Reading {filename_actual[i]} \n")
        actual[i][:] = rfu.read_array_from_file(os.path.join(actual_diff_path, filename_actual[i]))
        logging.info(f"Reading {filename_autodiff[i]} \n")
        autodiff[i][:] = rfu.read_array_from_file(os.path.join(auto_diff_path, filename_autodiff[i]))
    
    # Compute the error
    
    error = actual - autodiff
    plot_data(error[0][:],"domega_dot_1_dY2", error_path,save_plot=False)
    # print("Max error for domega_dot_1_drho: ", np.max(np.abs(error[0][:])))
    # print("Max error for domega_dot_1_dY2: ", np.max(np.abs(error[1][:])))
    



    # x = np.arange(1, (error.shape[1] + 1), 1)
    # Plot the error for each variable
  
    # plt.plot(x, -np.log10(np.abs(error[0][:])), label='CN_rho', color='green')
    # plt.plot(x, -np.log10(np.abs(error[1][:])), label='CN_T', color='magenta')
    # plt.plot(x, -np.log10(np.abs(error[2][:])), label='CN_Y1', color='blue')
    # plt.plot(x, -np.log10(np.abs(error[3][:])), label='CN_Y2', color='orange')
    # plt.plot(x, -np.log10(np.abs(error[4][:])), label='CN_Y3', color='red')
    # plt.plot(x, -np.log10(np.abs(error[5][:])), label='CN_Y4', color='purple')
    # plt.xlabel('n')
    # plt.ylabel('-log10(|Error|)')
    # title = f"CN fields Computation error"
    # plt.title(title)
    # plt.grid(True)
    # # plt.xlim(0, 100)
    # plt.legend()
    # plt.show()
    # plt.savefig(os.path.join(error_path, f"error_wrt_{var[i]}.png"))
    
    # plt.close()
     

try:
    if __name__ == "__main__":
        main()  # Call the main function
except Exception as e:
    logging.error("An error occurred during execution", exc_info=True)