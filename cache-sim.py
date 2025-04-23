"""
This script simulates a cache system with a given number of sets and blocks per set.
"""

import numpy as np
import argparse
from cpu_double import CPU
from memory_double import Memory, Cache
from algorithims import daxpy, mxm, mxm_tiled
import math

DOUBLE_SIZE = 8

def main(args):
    args.b //= DOUBLE_SIZE # Convert to number of doubles

    if args.t:
        if args.a == "daxpy":
            args.d = 9
        elif args.a == "mxm":
            args.d = 3

    if args.a == "daxpy":
        # args.d memory is needed to store each of the 3 vectors with 1 extra double needed for the scalar
        memsize = DOUBLE_SIZE * math.ceil((3 * args.d + 1) / args.b) * args.b
    else:
        # args.d * args.d memory is needed to store each of the 3 matrices
        memsize = DOUBLE_SIZE * math.ceil(3 * args.d * args.d / args.b) * args.b
    
    # Ensures that there is at least 1 tag bit. TODO change to 0 bits
    memsize = max(memsize, args.c)
    
    mem = Memory(memsize)
    assert args.c % (DOUBLE_SIZE * args.b * args.n) == 0, "Cache size must be a multiple of the block size, associativity, and word size."
    total_blocks = args.c // (args.b * DOUBLE_SIZE)
    num_sets = total_blocks // args.n
    cache = Cache(
        num_sets=num_sets,
        block_size=args.b,
        associativity=args.n,
        replacement_policy=args.r,
        memory=mem
    )
    cpu = CPU(mem, cache)
    if args.a == "daxpy":
        if args.t:
            a = np.float64(3.0)
            x = np.array(list(range(9)), dtype=np.float64)
            y = np.array(list(range(0, 18, 2)), dtype=np.float64)
        else:
            a, x, y = None, None, None
        results = daxpy(cpu, args.d, args.p, a, x, y)
    elif args.a == "mxm":
        if args.t:
            A = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.float64)
            B = np.array([[0, 2, 4], [6, 8, 10], [12, 14, 16]], dtype=np.float64)
        else:
            A, B = None, None
        results = mxm(cpu, args.d, args.p, A, B)
    elif args.a == "mxm_block":
        if args.t:
            A = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.float64)
            B = np.array([[0, 2, 4], [6, 8, 10], [12, 14, 16]], dtype=np.float64)
        else:
            A, B = None, None
        results = mxm_tiled(cpu, args.d, args.f, args.p, A, B)
    
    print("INPUTS====================================")
    print(f"Ram Size =                          {memsize} bytes")
    print(f"Cache Size =                        {args.c} bytes")
    print(f"Block Size =                        {args.b * DOUBLE_SIZE} bytes")
    print(f"Word Size =                         {DOUBLE_SIZE} bytes")
    print(f"Total Blocks in Cache =             {args.c // (args.b * DOUBLE_SIZE)}")
    print(f"Associativity =                     {args.n}")
    print(f"Number of Sets =                    {num_sets}")
    print(f"Replacement Policy =                {args.r}")
    print(f"Algorithm =                         {args.a}")
    print(f"MXM Blocking Factor =               {args.f}")
    print(f"Matrix or Vector dimension =        {args.d}")
    print("RESULTS====================================")
    print(f"Instruction count:                  {results['instruction_count']}")
    print(f"Read hits:                          {results['load_hits']}")
    print(f"Read misses:                        {results['load_misses']}")
    print(f"Read miss rate:                     {results['load_misses'] / results['loads']:.2%}")
    print(f"Write hits:                         {results['write_hits']}")
    print(f"Write misses:                       {results['write_misses']}")
    print(f"Write miss rate:                    {results['write_misses'] / results['writes']:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="cache-sim",
        description="Simulate a cache system with a given number of sets and blocks per set."
    )

    parser.add_argument(
        "-c", help="The size of the cache in bytes.", type=int, default=65536
    )

    parser.add_argument(
        "-b", help="The size of a data block in bytes", type=int, default=64
    )

    parser.add_argument(
        "-n", help="The n-way associativity of the cache.", type=int, default=2
    )

    parser.add_argument(
        "-r", help="The replacement policy", type=str, default="LRU", choices=["LRU", "FIFO", "random"]
    )

    parser.add_argument(
        "-a", help="The algorithm to simulate", type=str, default="mxm_block", choices=["daxpy", "mxm", "mxm_block"]
    )

    parser.add_argument(
        "-d", help="The dimension of the algorithmic matrix (or vector) operation.", type=int, default=480
    )

    parser.add_argument(
        "-p", help="Enables printing of the resulting “solution” matrix product or daxpy vector", action="store_true"
    )

    parser.add_argument(
        "-f", help="The blocking factor for use when using the blocked matrix multiplication algorithm.", type=int, default=32
    )

    parser.add_argument(
        "-t", help="Whether to use test initialization for vectors/matracies", action="store_true"
    )

    args = parser.parse_args()

    main(args)
