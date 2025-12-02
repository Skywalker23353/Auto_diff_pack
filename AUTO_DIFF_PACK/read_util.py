import jax.numpy as jnp
import numpy as np
# --------------Fucntion to read input file-------------------
def read_coords_from_file(filename):
    print("Reading input file:",filename)
    with open(filename, 'r') as file:
        array = jnp.array([float(line.strip()) for line in file], dtype=jnp.float64)
    print("File reading done")
    return array

def read_array_from_file(filename):
    print("Reading input file:",filename)
    with open(filename, 'r') as file:
        length = int(file.readline().strip())
        array = jnp.array([float(file.readline().strip()) for _ in range(length)], dtype=jnp.float64)
    print("File reading done")
    return array

def read_array_from_file_numpy(filename):
    print("Reading input file:",filename)
    with open(filename, 'r') as file:
        length = int(file.readline().strip())
        array = np.array([float(file.readline().strip()) for _ in range(length)], dtype=np.float64)
    print("File reading done")
    return array
#--------------------------------------------------------------------------------------
