"""
This script simulates a cache system with a given number of sets and blocks per set.
"""

import numpy as np
import argparse
from cpu import CPU
from memory import Memory, Cache
from algorithims import daxpy, mxm, mxm_tiled
import math

def main(args):
    if args.a == "daxpy":
        # args.d memory is needed to store each of the 3 vectors with 1 extra double needed for the scalar
        memsize = math.ceil((3 * args.d + 1) * 8 / args.b) * args.b
    else:
        # args.d * args.d memory is needed to store each of the 3 matrices
        memsize = math.ceil(3 * args.d * args.d * 8 / args.b) * args.b
    
    mem = Memory(memsize)
    assert args.c % (args.b * args.n) == 0, "Cache size must be a multiple of the block size and associativity."
    num_sets = args.c // (args.b * args.n)
    cache = Cache(
        num_sets=num_sets,
        block_size=args.b,
        associativity=args.n,
        replacement_policy=args.r,
        memory=mem
    )
    cpu = CPU(mem, cache)
    if args.a == "daxpy":
        results = daxpy(cpu, args.d, args.p)
    elif args.a == "mxm":
        results = mxm(cpu, args.d, args.p)
    elif args.a == "mxm_block":
        results = mxm_tiled(cpu, args.d, args.f, args.p)
    
    print("INPUTS====================================")
    print(f"Ram Size =                          {memsize} bytes")
    print(f"Cache Size =                        {args.c} bytes")
    print(f"Block Size =                        {args.b} bytes")
    print(f"Total Blocks in Cache =             {args.c // args.b}")
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
        "-p", help="Enables printing of the resulting “solution” matrix product or daxpy vector", type=bool, default=False
    )

    parser.add_argument(
        "-f", help="The blocking factor for use when using the blocked matrix multiplication algorithm.", type=int, default=32
    )

    args = parser.parse_args()

    main(args)
