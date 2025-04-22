import numpy as np
from cpu import CPU
from memory import Memory, Cache

def daxpy(cpu: CPU, n: int, print: bool) -> dict:
    """
    Performs the DAXPY operation: z = a * x + y, where a is a scalar,
    x, y, and z are vectors of length n.

    Args:
        cpu (CPU): The CPU instance.
        n (int): The length of the vectors.
        print(bool): Whether to print the results.
    
    Returns:
        dict: A dictionary containing the results and performance metrics.
    """

    a = np.random.rand()
    x = np.random.rand(n)
    y = np.random.rand(n)
    z = np.zeros(n)

    intended_z = (x * a) + y

    # a is stored at address 0
    # x is stored at addresses 1 to n
    # y is stored at addresses n + 1 to 2n
    # z is stored at addresses 2n + 1 to 3n
    cpu.storeDouble(0, a)
    for i in range(1, n + 1):
        cpu.storeDouble(i, x[i - 1])
        cpu.storeDouble(n + i, y[i - 1])
    
    for i in range(1, n + 1):
        z[i-1] = cpu.loadFMADouble(0, i, n + i, 2 * n + i)
        assert np.isclose(z[i-1], intended_z[i - 1]), f"z[{i - 1}] = {z[i-1]}, intended z[{i}] = {intended_z[i - 1]}"
    
    if print:
        for i in range(n):
            z[i] = cpu.loadDouble(2 * n + i + 1)
        print(f"z:\n{z}")
    
    results = {
        "loads" : cpu.cache.get_loads,
        "load_hits" : cpu.cache.get_load_hits,
        "load_misses" : cpu.cache.get_load_misses,
        "writes" : cpu.memory.get_writes,
        "write_hits" : cpu.cache.get_write_hits,
        "write_misses" : cpu.cache.get_write_misses,
        "instruction_count" : cpu.instruction_count,
        "a" : a,
        "x" : x,
        "y" : y,
        "z" : z,
        "intended_z" : intended_z
    }

    return results

def mxm(cpu: CPU, n: int, print: bool) -> dict:
    """
    Performs the matrix multiplication operation: C = A * B, where A, B, and C are matrices of size n x n.

    Args:
        cpu (CPU): The CPU instance.
        n (int): The size of the matrices.
        print(bool): Whether to print the results.
    
    Returns:
        dict: A dictionary containing the results and performance metrics.
    """

    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    C = np.zeros((n, n))

    intended_C = np.dot(A, B)

    # A is stored at addresses 0 to n^2
    # B is stored at addresses n^2 to 2n^2
    # C is stored at addresses 2n^2 to 3n^2
    for i in range(n):
        for j in range(n):
            cpu.storeDouble(i * n + j, A[i][j])
            cpu.storeDouble(n * n + i * n + j, B[i][j])
            cpu.storeDouble(2 * n * n + i * n + j, C[i][j])
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                addressA = i * n + k
                addressB = n * n + k * n + j
                addressC = 2 * n * n + i * n + j
                C[i][j] += cpu.loadFMADouble(addressA, addressB, addressC, addressC)
            assert np.isclose(C[i][j], intended_C[i][j]), f"C[{i}][{j}] = {C[i][j]}, intended C[{i}][{j}] = {intended_C[i][j]}"
    
    if print:
        for i in range(n):
            for j in range(n):
                C[i][j] = cpu.loadDouble(2 * n * n + i * n + j)
        print(f"C:\n{C}")
    
    results = {
        "loads" : cpu.cache.get_loads,
        "load_hits" : cpu.cache.get_load_hits,
        "load_misses" : cpu.cache.get_load_misses,
        "writes" : cpu.memory.get_writes,
        "write_hits" : cpu.cache.get_write_hits,
        "write_misses" : cpu.cache.get_write_misses,
        "instruction_count" : cpu.instruction_count,
        "A" : A,
        "B" : B,
        "C" : C,
        "intended_C" : intended_C
    }

    return results

def mxm_tiled(cpu: CPU, n: int, tile_size: int) -> dict:
    """
    Performs the matrix multiplication operation: C = A * B, where A, B, and C are matrices of size n x n.
    This version uses tiling to optimize memory access patterns.

    Args:
        cpu (CPU): The CPU instance.
        n (int): The size of the matrices.
        tile_size (int): The size of the blocks for tiling.
    
    Returns:
        dict: A dictionary containing the results and performance metrics.
    """

    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    C = np.zeros((n, n))

    intended_C = np.dot(A, B)

    # A is stored at addresses 0 to n^2
    # B is stored at addresses n^2 to 2n^2
    # C is stored at addresses 2n^2 to 3n^2
    for i in range(n):
        for j in range(n):
            cpu.storeDouble(i * n + j, A[i][j])
            cpu.storeDouble(n * n + i * n + j, B[i][j])
            cpu.storeDouble(2 * n * n + i * n + j, C[i][j])
    
    for i in range(0, n, tile_size):
        for j in range(0, n, tile_size):
            for k in range(0, n, tile_size):
                for ii in range(i, min(i + tile_size, n)):
                    for jj in range(j, min(j + tile_size, n)):
                        for kk in range(k, min(k + tile_size, n)):
                            addressA = ii * n + kk
                            addressB = n * n + kk * n + jj
                            addressC = 2 * n * n + ii * n + jj
                            C[ii][jj] += cpu.loadFMADouble(addressA, addressB, addressC, addressC)
            for ii in range(i, min(i + tile_size, n)):
                for jj in range(j, min(j + tile_size, n)):
                    assert np.isclose(C[ii][jj], intended_C[ii][jj]), f"C[{ii}][{jj}] = {C[ii][jj]}, intended C[{ii}][{jj}] = {intended_C[ii][jj]}"
    
    if print:
        for i in range(n):
            for j in range(n):
                C[i][j] = cpu.loadDouble(2 * n * n + i * n + j)
        print(f"C:\n{C}")

    results = {
        "loads" : cpu.cache.get_loads,
        "load_hits" : cpu.cache.get_load_hits,
        "load_misses" : cpu.cache.get_load_misses,
        "writes" : cpu.memory.get_writes,
        "write_hits" : cpu.cache.get_write_hits,
        "write_misses" : cpu.cache.get_write_misses,
        "instruction_count" : cpu.instruction_count,
        "A" : A,
        "B" : B,
        "C" : C,
        "intended_C" : intended_C
    }

    return results
