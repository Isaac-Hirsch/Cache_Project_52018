import numpy as np
import pytest

from memory import Memory, Cache
from cpu import CPU, DOUBLE_SIZE   # DOUBLE_SIZE == 8

###############################################################################
# Helpers
###############################################################################
def write_double_direct(mem: Memory, addr: int, value: float) -> None:
    """
    Bypass the CPU/cache hierarchy and stuff `value` directly into main memory.
    """
    raw = np.array([value], dtype=np.float64).view(np.ubyte)
    for i in range(DOUBLE_SIZE):
        mem.write(addr + i, np.ubyte(raw[i]))


###############################################################################
# Pytest fixtures
###############################################################################
@pytest.fixture
def arch():
    """
    Fresh CPU + Memory + Cache for every test.
    256 B memory, 4 B blocks, 2‑way, 4 sets, LRU.
    """
    mem   = Memory(256)
    cache = Cache(num_sets=4, block_size=4, associativity=2,
                  replacement_policy="LRU", memory=mem)
    cpu   = CPU(mem, cache)
    return cpu, mem, cache


###############################################################################
# Basic load / store
###############################################################################
@pytest.mark.parametrize("addr", [0, 3])        # aligned and unaligned
def test_load_store_double_correctness(arch, addr):
    cpu, mem, cache = arch
    value = np.pi

    prev_mem_w, prev_cache_w = mem.get_writes, cache.get_writes

    cpu.storeDouble(addr, value)
    loaded = cpu.loadDouble(addr)

    assert np.isclose(loaded, value, rtol=0, atol=1e-12)
    assert mem.get_writes   - prev_mem_w   == DOUBLE_SIZE
    assert cache.get_writes - prev_cache_w == DOUBLE_SIZE
    assert cpu.getInstructionCount == 2


###############################################################################
# Pure ALU ops
###############################################################################
def test_add_and_mult_increment_ic_and_value(arch):
    cpu, *_ = arch
    assert np.isclose(cpu.addDouble(1.5, 2.5), 4.0)
    assert cpu.getInstructionCount == 1

    assert np.isclose(cpu.multDoubles(3.0, -2.0), -6.0)
    assert cpu.getInstructionCount == 2     # +1 more


###############################################################################
# Compound instructions – data‑flow, IC, and side effects
###############################################################################
def test_load_add_double_sequence(arch):
    cpu, mem, cache = arch
    a_addr, b_addr, c_addr = 8, 16, 24
    write_double_direct(mem, a_addr, 1.25)
    write_double_direct(mem, b_addr, -0.75)

    start_r, start_w, start_cw = mem.get_reads, mem.get_writes, cache.get_writes
    cpu.loadAddDouble(a_addr, b_addr, c_addr)

    out = cpu.loadDouble(c_addr)
    assert np.isclose(out, 0.5, atol=1e-12)

    # Two loads → 16 reads, one store → 8 writes through cache
    assert mem.get_reads  - start_r  == 3 * DOUBLE_SIZE
    assert mem.get_writes - start_w  == DOUBLE_SIZE
    assert cache.get_writes - start_cw == DOUBLE_SIZE

    # Internally: load + load + add + store = 4 micro‑ops
    assert cpu.getInstructionCount == 5


def test_load_mult_double_sequence_returns_value(arch):
    cpu, mem, _ = arch
    a_addr, b_addr, c_addr = 32, 40, 48
    write_double_direct(mem, a_addr, 6.0)
    write_double_direct(mem, b_addr, 7.0)

    result = cpu.loadMultDouble(a_addr, b_addr, c_addr)
    assert np.isclose(result, 42.0)
    assert np.isclose(cpu.loadDouble(c_addr), 42.0)
    assert cpu.getInstructionCount == 5


def test_load_fma_double_sequence(arch):
    cpu, mem, _ = arch
    a_addr, b_addr, c_addr, d_addr = 56, 64, 72, 80
    write_double_direct(mem, a_addr, 2.0)
    write_double_direct(mem, b_addr, 3.0)
    write_double_direct(mem, c_addr, 10.0)

    res = cpu.loadFMADouble(a_addr, b_addr, c_addr, d_addr)
    assert np.isclose(res, 16.0)                         # (2*3)+10
    assert np.isclose(cpu.loadDouble(d_addr), 16.0)
    # 2 loads + mul + load + add + store  ==> 6
    assert cpu.getInstructionCount == 7


###############################################################################
# Cache‑level behaviour
###############################################################################
def test_cache_write_hits_on_second_store(arch):
    cpu, mem, cache = arch
    addr = 88
    val1, val2 = 1.0, 2.0

    cpu.storeDouble(addr, val1)
    hits_after_first = cache.get_write_hits
    cpu.storeDouble(addr, val2)

    # Second pass should be all hits (8 bytes)
    assert cache.get_write_hits - hits_after_first == DOUBLE_SIZE
    assert np.isclose(cpu.loadDouble(addr), val2)


###############################################################################
# Reset semantics
###############################################################################
def test_memory_and_cache_reset(arch):
    _, mem, cache = arch

    # Touch memory/cache
    cache.write(0, np.ubyte(123))
    mem.read(0)

    cache.cache_reset()
    mem.reset()

    assert mem.get_reads == 0 and mem.get_writes == 0
    assert cache.get_loads == 0 and cache.get_writes == 0
    assert not cache.valid_bits.any()

def test_cpu_byte_conversion(arch):
    cpu, mem, _ = arch
    addr = 96
    value = np.float64(3.14)

    # Store a double in memory
    cpu.storeDouble(addr, value)

    # Load the double back from memory
    loaded_value = cpu.loadDouble(addr)

    # Check if the loaded value is close to the original value
    assert np.isclose(loaded_value, value, rtol=0, atol=1e-12)
