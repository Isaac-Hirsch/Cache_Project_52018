import numpy as np
from cpu_double import CPU
from typing import Optional

def daxpy(
        cpu: CPU,
        n: int,
        print_results: bool,
        a: Optional[np.float64] = None,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        ) -> dict:
    """
    Performs the DAXPY operation: z = a * x + y, where a is a scalar,
    x, y, and z are vectors of length n.

    Args:
        cpu (CPU): The CPU instance.
        n (int): The length of the vectors.
        print_results(bool): Whether to print the results.
        a (Optional[np.float64]): The scalar. If None, a random value is generated.
        x (Optional[np.ndarray]): The first vector. If None, a random vector is generated.
        y (Optional[np.ndarray]): The second vector. If None, a random vector is generated.
    
    Returns:
        dict: A dictionary containing the results and performance metrics.
    """

    if a is None:
        a = np.float64(np.random.rand())
    if x is None:
        x = np.random.rand(n).astype(np.float64)
    if y is None:
        y = np.random.rand(n).astype(np.float64)

    assert len(x) == n, "x must be of length n"
    assert len(y) == n, "y must be of length n"
    assert isinstance(a, np.float64), "a must be a float"

    z = np.zeros(n, dtype=np.float64)

    intended_z = (x * a) + y

    # a is stored at address 0
    # x is stored at addresses 1 to n
    # y is stored at addresses n + 1 to 2n
    # z is stored at addresses 2n + 1 to 3n
    cpu.storeDouble(0, a)
    for i in range(1, n + 1):
        cpu.storeDouble((i),       x[i - 1])
        cpu.storeDouble((n + i),   y[i - 1])
    
    for i in range(1, n + 1):
        address_a = 0
        address_x = i
        address_y = (n + i)
        address_z = (2 * n + i)
        z[i-1] = cpu.loadFMADouble(address_a, address_x, address_y, address_z)
        assert np.isclose(z[i-1], intended_z[i - 1]), f"z[{i - 1}] = {z[i-1]}, intended z[{i}] = {intended_z[i - 1]}"
    
    if print_results:
        for i in range(n):
            address = (2 * n + i + 1)
            z[i] = cpu.loadDouble(address)
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

def mxm(
        cpu: CPU,
        n: int,
        print_results: bool,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        ) -> dict:
    """
    Performs the matrix multiplication operation: C = A * B, where A, B, and C are matrices of size n x n.

    Args:
        cpu (CPU): The CPU instance.
        n (int): The size of the matrices.
        print_results(bool): Whether to print the results.
        A (Optional[np.ndarray]): The first matrix. If None, a random matrix is generated.
        B (Optional[np.ndarray]): The second matrix. If None, a random matrix is generated.
    
    Returns:
        dict: A dictionary containing the results and performance metrics.
    """
    if A is None:
        A = np.random.rand(n, n).astype(np.float64)
    if B is None:
        B = np.random.rand(n, n).astype(np.float64)
    C = np.zeros((n, n), dtype=np.float64)

    assert A.shape == (n, n), "A must be of shape (n, n)"
    assert B.shape == (n, n), "B must be of shape (n, n)"

    intended_C = np.dot(A, B)

    # A is stored at addresses 0 to n^2
    # B is stored at addresses n^2 to 2n^2
    # C is stored at addresses 2n^2 to 3n^2
    for i in range(n):
        for j in range(n):
            cpu.storeDouble((i * n + j),              A[i][j])
            cpu.storeDouble((n * n + i * n + j),      B[i][j])
            cpu.storeDouble((2 * n * n + i * n + j),  C[i][j])
    
    for i in range(n):
        for j in range(n):
            addressC = (2 * n * n + i * n + j)
            temp = cpu.loadDouble(addressC)
            for k in range(n):
                addressA = (i * n + k)
                addressB = (n * n + k * n + j)
                A_ik = cpu.loadDouble(addressA)
                B_jk = cpu.loadDouble(addressB)
                mult_step = cpu.multDoubles(A_ik, B_jk)
                temp = cpu.addDoubles(temp, mult_step)
            C[i][j] = temp
            cpu.storeDouble(addressC, C[i][j])
            assert np.isclose(C[i][j], intended_C[i][j]), f"C[{i}][{j}] = {C[i][j]}, intended C[{i}][{j}] = {intended_C[i][j]}"
    
    if print_results:
        for i in range(n):
            for j in range(n):
                address = (2 * n * n + i * n + j)
                C[i][j] = cpu.loadDouble(address)
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

def mxm_tiled(
        cpu: CPU,
        n: int,
        tile_size: int,
        print_results: bool,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        ) -> dict:
    """
    Performs the matrix multiplication operation: C = A * B, where A, B, and C are matrices of size n x n.
    This version uses tiling to optimize memory access patterns.

    Args:
        cpu (CPU): The CPU instance.
        n (int): The size of the matrices.
        tile_size (int): The size of the blocks for tiling.
        print_results(bool): Whether to print the results.
        A (Optional[np.ndarray]): The first matrix. If None, a random matrix is generated.
        B (Optional[np.ndarray]): The second matrix. If None, a random matrix is generated.
    
    Returns:
        dict: A dictionary containing the results and performance metrics.
    """

    if A is None:
        A = np.random.rand(n, n).astype(np.float64)
    if B is None:
        B = np.random.rand(n, n).astype(np.float64)
    C = np.zeros((n, n), dtype=np.float64)
    
    assert A.shape == (n, n), "A must be of shape (n, n)"
    assert B.shape == (n, n), "B must be of shape (n, n)"

    intended_C = np.dot(A, B)

    # A is stored at addresses 0 to n^2
    # B is stored at addresses n^2 to 2n^2
    # C is stored at addresses 2n^2 to 3n^2
    for i in range(n):
        for j in range(n):
            cpu.storeDouble((i * n + j),              A[i][j])
            cpu.storeDouble((n * n + i * n + j),      B[i][j])
            cpu.storeDouble((2 * n * n + i * n + j),  C[i][j])
    
    for i in range(0, n, tile_size):
        for j in range(0, n, tile_size):
            for k in range(0, n, tile_size):
                for ii in range(i, min(i + tile_size, n)):
                    for jj in range(j, min(j + tile_size, n)):
                        addressC = (2 * n * n + ii * n + jj)
                        temp = cpu.loadDouble(addressC)
                        for kk in range(k, min(k + tile_size, n)):
                            addressA = (ii * n + kk)
                            addressB = (n * n + kk * n + jj)
                            A_ii_kk = cpu.loadDouble(addressA)
                            B_jj_kk = cpu.loadDouble(addressB)
                            mult_step = cpu.multDoubles(A_ii_kk, B_jj_kk)
                            temp = cpu.addDoubles(temp, mult_step)
                        C[ii][jj] = temp
                        cpu.storeDouble(addressC, C[ii][jj])
         
            for ii in range(i, min(i + tile_size, n)):
                for jj in range(j, min(j + tile_size, n)):
                    assert np.isclose(C[ii][jj], intended_C[ii][jj]), f"C[{ii}][{jj}] = {C[ii][jj]}, intended C[{ii}][{jj}] = {intended_C[ii][jj]}"
    
    if print_results:
        for i in range(n):
            for j in range(n):
                address = (2 * n * n + i * n + j)
                C[i][j] = cpu.loadDouble(address)
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
