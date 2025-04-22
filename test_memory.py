import pytest
import numpy as np
import math

from memory import Memory, Cache


#
# ——— MEMORY TESTS ———
#

def test_memory_initialization():
    m = Memory(memory_size=32)
    assert m.memory_size == 32
    # underlying numpy array should be zeros
    assert np.all(m.memory == 0)
    # address_size = ceil(log2(32)) == 5
    assert m.get_address_size == math.ceil(math.log2(32))
    assert m.get_reads == 0
    assert m.get_writes == 0

@pytest.mark.parametrize("addr", [-1, 32, 100])
def test_memory_read_out_of_bounds(addr):
    m = Memory(32)
    with pytest.raises(AssertionError):
        m.read(addr)

@pytest.mark.parametrize("addr", [-5, 32, 100])
def test_memory_write_out_of_bounds(addr):
    m = Memory(32)
    with pytest.raises(AssertionError):
        m.write(addr, np.ubyte(0))

def test_memory_write_type_check():
    m = Memory(16)
    # writing an int (not np.ubyte) should fail
    with pytest.raises(AssertionError):
        m.write(0, 123)

def test_memory_read_write_and_counters():
    m = Memory(8)
    # write a byte, then read it back
    data = np.ubyte(42)
    m.write(3, data)
    assert m.get_writes == 1
    assert m.read(3) == data
    assert m.get_reads == 1

def test_memory_reset():
    m = Memory(8)
    m.write(2, np.ubyte(7))
    m.read(2)
    assert m.get_writes == 1 and m.get_reads == 1
    m.reset()
    assert m.get_reads == 0 and m.get_writes == 0
    assert np.all(m.memory == 0)


#
# ——— CACHE TESTS ———
#

def test_cache_init_invalid_params():
    mem = Memory(16)
    with pytest.raises(AssertionError):
        Cache(num_sets=0, block_size=4, associativity=1, replacement_policy="LRU", memory=mem)
    with pytest.raises(AssertionError):
        Cache(num_sets=4, block_size=0, associativity=1, replacement_policy="LRU", memory=mem)
    with pytest.raises(AssertionError):
        Cache(num_sets=4, block_size=4, associativity=0, replacement_policy="LRU", memory=mem)
    with pytest.raises(AssertionError):
        Cache(num_sets=4, block_size=4, associativity=1, replacement_policy="NOTPOLICY", memory=mem)

def test_get_address_split_and_bounds():
    # Choose memory_size=256 so address_size=8 bits
    mem = Memory(256)
    # num_sets=4 -> index_size=2, block_size=4 -> offset_size=2, so tag_size=8-2-2=4
    cache = Cache(num_sets=4, block_size=4, associativity=1, replacement_policy="LRU", memory=mem)
    # Pick an address with known bit pattern: 0b1011_01_11 = 0xBB = 187
    addr = 0b10110111
    tag, idx, off = cache.get_address_split(addr)
    assert tag == 0b1011        # high 4 bits
    assert idx == 0b01          # next 2 bits
    assert off == 0b11          # low 2 bits
    # Out-of-bounds
    with pytest.raises(AssertionError):
        cache.get_address_split(256)

@pytest.mark.parametrize("policy", ["LRU", "FIFO", "Random"])
def test_cache_read_miss_and_hit_direct_mapped(policy):
    # Direct-mapped: num_sets=4, assoc=1, block_size=2
    mem = Memory(16)
    # fill memory 0..7 with unique bytes
    for i in range(16):
        mem.memory[i] = np.ubyte(i * 5)
    cache = Cache(num_sets=4, block_size=2, associativity=1, replacement_policy=policy, memory=mem)

    # First read of address=2 -> miss
    block, offset = cache.read(2)
    assert cache.get_loads == 1
    assert cache.get_load_hits == 0
    # It should have loaded 2 bytes: at 2 and 3
    assert mem.get_reads == 2
    assert offset == (2 & (2**cache.offset_size - 1))
    assert np.array_equal(block, np.array([10, 15], dtype=np.ubyte))

    # Second read of address=3 -> same block, so hit
    block2, offset2 = cache.read(3)
    assert cache.get_loads == 2
    assert cache.get_load_hits == 1
    # no further memory reads
    assert mem.get_reads == 2
    assert np.array_equal(block2, block)
    # For FIFO, ensure the pointer advanced on the hit
    if policy == "FIFO":
        assert cache.fifo[0] == 0
        assert cache.fifo[1] == 0
        assert cache.fifo[2] == 0
        assert cache.fifo[3] == 0

@pytest.mark.parametrize("policy", ["LRU"])
def test_lru_replacement_counters(policy):
    # 2-way, single set to force eviction
    mem = Memory(4)
    for i in range(4):
        mem.memory[i] = np.ubyte(i + 1)
    cache = Cache(num_sets=1, block_size=1, associativity=2, replacement_policy="LRU", memory=mem)

    # load addr 0,1 => two misses
    cache.read(0)
    assert list(cache.lru) == [0, 0]
    cache.read(1)
    # after loading 1, line0 gets bumped
    assert list(cache.lru) == [1, 0]

    # re‑access 0 => hit, should now be most recent (lru[0]=0), other bumped
    cache.read(0)
    assert list(cache.lru) == [0, 1]

def test_random_policy_basic():
    # 2-way, single set; block_size=1
    mem = Memory(4)
    # give memory non-zero values
    for i in range(4):
        mem.memory[i] = np.ubyte((i+1)*5)
    cache = Cache(num_sets=1, block_size=1, associativity=2, replacement_policy="Random", memory=mem)

    # two different addresses must fill both lines before eviction
    cache.read(0)
    cache.read(1)
    assert cache.get_loads == 2
    assert cache.get_load_hits == 0
    # both lines should now be valid
    assert all(cache.valid_bits)
    assert set(cache.tags) == {0, 1}

def test_cache_write_through_and_counters():
    # direct-mapped, block_size=2
    mem = Memory(8)
    cache = Cache(num_sets=4, block_size=2, associativity=1, replacement_policy="LRU", memory=mem)

    # first write -> miss (write-allocate), writes=1, write_hits=0
    cache.write(4, 80)
    cache.write(5, 110)
    assert cache.get_writes == 2
    assert cache.get_write_hits == 1
    # memory should have received 2 writes
    assert mem.get_writes == 2
    # now reading address 4 => hit
    _, _ = cache.read(4)
    assert cache.get_loads == 1
    # second write to same block -> hit
    cache.write(4, 80)
    assert cache.get_writes == 3
    assert cache.get_write_hits == 2
    assert mem.get_writes == 3  # two more byte‑writes

def test_cache_reset():
    mem = Memory(16)
    cache = Cache(num_sets=4, block_size=2, associativity=2, replacement_policy="LRU", memory=mem)
    cache.read(0)
    cache.write(4, 2)
    assert cache.get_loads > 0 or cache.get_writes > 0
    cache.cache_reset()
    # counters zeroed
    assert cache.get_loads == cache.get_load_hits == cache.get_writes == cache.get_write_hits == 0
    # all tags invalid and data zeroed
    assert not cache.valid_bits.any()
    assert np.all(cache.tags == 0)
    assert np.all(cache.data == 0)
    # LRU state zeroed
    if hasattr(cache, "lru"):
        assert np.all(cache.lru == 0)
    if hasattr(cache, "fifo"):
        assert np.all(cache.fifo == 0)
