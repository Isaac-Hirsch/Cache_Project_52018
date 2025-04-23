"""
This module implements a simple memory and cache system as described in project
1 of MPCS Advanced Computer Architecture.
"""

import numpy as np
import math
from typing import Tuple
try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

DOUBLE_SIZE = 8

@jit(nopython=True, cache=True)
def associative_tag_search(
            tags: np.ndarray,
            valid_bits: np.ndarray,
            tag: int,
            index: int,
            associativity: int
        ) -> Tuple[np.ndarray, int]:
    """
    Searches for a tag in the given index of the cache.
    
    Args:
        tags (np.ndarray): Array of tags.
        valid_bits (np.ndarray): Array of valid bits.
        tag (int): Tag to search for.
        index (int): Index to search in.
        associativity (int): Associativity of the cache.

    Returns:
        tuple[np.ndarray, int]: A tuple containing the result of the search and the index of the tag.
    """
    associativity_check = valid_bits[index * associativity:(index + 1) * associativity] & \
                (tags[index * associativity:(index + 1) * associativity] == tag)
    associative_index = np.argmax(associativity_check)
    return associativity_check[associative_index], associative_index

@jit(nopython=True, cache=True)
def LRU_update(
            lru: np.ndarray,
            cache_line: int,
            index: int,
            associativity: int,
            valid_bits: np.ndarray
        ):
    """
    Updates the LRU array for the given cache line.

    Args:
        lru (np.ndarray): LRU array.
        cache_line (int): Cache line to update.
        index (int): Index of the cache line.
        associativity (int): Associativity of the cache.
        valid_bits (np.ndarray): Array of valid bits.
    
    Returns:
        None
    """
    lru[index * associativity:(index + 1) * associativity] \
        += valid_bits[index * associativity:(index + 1) * associativity] & \
        (lru[index * associativity:(index + 1) * associativity] < lru[cache_line])
    lru[cache_line] = 0

class Memory:
    def __init__(self, memory_size: int):
        """
        Initializes the memory with the given size.
        This memory is word-addressable with a word size of 8 bytes.

        Args:
            memory_size (int): Size of the memory in bytes.
        """
        self.memory_size = memory_size // DOUBLE_SIZE
        self.memory = np.zeros((self.memory_size, ), dtype=np.float64)
        self.address_size = math.ceil(np.log2(self.memory_size))

        self.reads = 0
        self.writes = 0

    def read(self, address: int) -> np.float64:
        """"
        Reads a float from the memory at the given address.
        """
        assert 0 <= address < self.memory_size, "Address out of bounds"
    
        self.reads += 1
        return self.memory[address]
    
    def write(self, address: int, data: np.float64) -> None:
        """
        Writes a ibyte to the memory at the given address.
        """
        assert 0 <= address < self.memory_size, "Address out of bounds"
        assert isinstance(data, np.float64), "Data must be of type np.float64"
    
        self.memory[address] = data

        self.writes += 1

    def reset(self) -> None:
        self.memory.fill(0)
        self.reads = 0
        self.writes = 0
    
    @property
    def get_reads(self) -> int:
        return self.reads
    
    @property
    def get_writes(self) -> int:
        return self.writes
    
    @property
    def get_address_size(self) -> int:
        return self.address_size
    
class Cache:
    def __init__(
                    self,
                    num_sets: int,
                    block_size: int,
                    associativity: int,
                    replacement_policy: str,
                    memory: Memory
                ):
        """
        Initializes the cache with the given parameters.

        Args:
            num_sets (int): Number of sets in the cache.
            block_size (int): Size of each block in bytes.
            associativity (int): Associativity of the cache.
            replacement_policy (str): Replacement policy to use ("LRU", "FIFO", "Random").
            memory (Memory): Memory object to interact with.
        """
        assert block_size > 0, "Block size must be greater than 0"
        assert associativity > 0, "Associativity must be greater than 0"
        assert num_sets > 0, "Cache lines must be greater than 0"
        assert replacement_policy in ["LRU", "FIFO", "random"], "Invalid replacement policy"
    
        self.block_size = block_size
        self.associativity = associativity
        self.replacement_policy = replacement_policy
        self.cache_size = num_sets * block_size * associativity
        self.cache_lines = num_sets * associativity
        self.num_sets = num_sets

        self.memory = memory

        self.index_size = math.ceil(np.log2(self.num_sets))
        self.offset_size = math.ceil(np.log2(block_size))
        self.tag_size = memory.get_address_size - self.index_size - self.offset_size
        assert self.tag_size >= 0, f"Tag size must be greater than or equal to 0. Got {self.tag_size}, index size {self.index_size}, offset size {self.offset_size}, address size {memory.get_address_size}"
        #This does not include bits used for replacement policy
        self.cache_line_size = self.block_size + self.tag_size + 1 # Valid bit
        
        self.tags = np.zeros((self.cache_lines,), dtype=np.int64)
        self.valid_bits = np.zeros((self.cache_lines, ), dtype=np.bool_)
        self.data = np.zeros((self.cache_lines, block_size), dtype=np.float64)

        if replacement_policy == "LRU":
            self.lru = np.zeros((self.cache_lines, ), dtype=np.int32)
        
        if replacement_policy == "FIFO":
            self.fifo = np.zeros((self.num_sets, ), dtype=np.int32)
        
        self.loads = 0
        self.load_hits = 0

        self.writes = 0
        self.write_hits = 0 #Not uses in this implementation since we are implementing write-through cache

        self._index_mask = 2 ** self.index_size - 1
        self._offset_mask = 2 ** self.offset_size - 1
        self._associativity_check = np.zeros((self.associativity, ), dtype=np.bool_)
    
    def get_address_split(self, address: int) -> Tuple[int, int, int]:
        """
        Splits the address into tag, index, and offset.
        """
        assert 0 <= address < self.memory.memory_size, "Address out of bounds"
    
        tag = address >> (self.index_size + self.offset_size)
        index = (address >> self.offset_size) & self._index_mask
        offset = address & self._offset_mask

        return tag, index, offset
    
    def read(self, address: int) -> Tuple[np.float64, int]:
        assert 0 <= address < self.memory.memory_size, "Address out of bounds"
    
        self.loads += 1

        tag, index, offset = self.get_address_split(address)
        block_address = address - offset

        cache_hit, associative_index = associative_tag_search(
            tags=self.tags,
            valid_bits=self.valid_bits,
            tag=tag,
            index=index,
            associativity=self.associativity
        )
        if cache_hit:
            # Cache hit
            cache_line = index * self.associativity + associative_index
            self.load_hits += 1
            if self.replacement_policy == "LRU" and self.lru[cache_line]:
                LRU_update(
                    lru=self.lru,
                    cache_line=cache_line,
                    index=index,
                    associativity=self.associativity,
                    valid_bits=self.valid_bits
                )
            return self.data[cache_line], offset
        # Cache miss
        if self.replacement_policy == "LRU":
            self._associativity_check[:] = 1 - self.valid_bits[index * self.associativity:(index + 1) * self.associativity] | \
                  self.lru[index * self.associativity:(index + 1) * self.associativity] == self.associativity - 1
            associative_index = np.argmax(self._associativity_check)
            cache_line = index * self.associativity + associative_index
            self.valid_bits[cache_line] = True
            self.tags[cache_line] = tag

            for j in range(self.block_size):
                self.data[cache_line][j] = self.memory.read(block_address + j)
            
            self.lru[index * self.associativity:(index + 1) * self.associativity] \
                += self.valid_bits[index * self.associativity:(index + 1) * self.associativity]
            self.lru[cache_line] = 0
            
            return self.data[cache_line], offset

        elif self.replacement_policy == "FIFO":
            cache_line = index * self.associativity + self.fifo[index]
            self.valid_bits[cache_line] = True
            self.tags[cache_line] = tag
            for j in range(self.block_size):
                self.data[cache_line][j] = self.memory.read(block_address + j)
            self.fifo[index] = (self.fifo[index] + 1) % self.associativity
            return self.data[cache_line], offset
        
        elif self.replacement_policy == "random":
            self._associativity_check[:] = 1 - self.valid_bits[index * self.associativity:(index + 1) * self.associativity]
            associative_index = np.argmax(self._associativity_check)
            if self._associativity_check[associative_index]:
                cache_line = index * self.associativity + associative_index
                if not self.valid_bits[cache_line]:
                    self.valid_bits[cache_line] = True
                    self.tags[cache_line] = tag

                    for j in range(self.block_size):
                        self.data[cache_line][j] = self.memory.read(block_address + j)
                    
                    return self.data[cache_line], offset
            else:
                cache_line = index * self.associativity + np.random.randint(self.associativity)
                self.valid_bits[cache_line] = True
                self.tags[cache_line] = tag
                for j in range(self.block_size):
                    self.data[cache_line][j] = self.memory.read(block_address + j)
                return self.data[cache_line], offset
    
    def write(self, address: int, data: np.float64) -> None:
        """
        Writes a block of data using a write-through policy.

        Args:
            address (int): Address to write to.
            data (np.float64): Data to write.
        """
        assert 0 <= address < self.memory.memory_size, "Address out of bounds"
        data = np.float64(data)
    
        self.writes += 1

        tag, index, offset = self.get_address_split(address)
        block_address = address - offset

        cache_hit, associative_index = associative_tag_search(
            tags=self.tags,
            valid_bits=self.valid_bits,
            tag=tag,
            index=index,
            associativity=self.associativity
        )
        if cache_hit:
            # Cache hit
            cache_line = index * self.associativity + associative_index
            self.write_hits += 1

            self.data[cache_line][offset] = data
            
            if self.replacement_policy == "LRU" and self.lru[cache_line]:
                LRU_update(
                    lru=self.lru,
                    cache_line=cache_line,
                    index=index,
                    associativity=self.associativity,
                    valid_bits=self.valid_bits
                )
        else:
            # Cache miss
            if self.replacement_policy == "LRU":
                self._associativity_check[:] = 1 - self.valid_bits[index * self.associativity:(index + 1) * self.associativity] | \
                    self.lru[index * self.associativity:(index + 1) * self.associativity] == self.associativity - 1
                associative_index = np.argmax(self._associativity_check)
                cache_line = index * self.associativity + associative_index
                self.valid_bits[cache_line] = True
                self.tags[cache_line] = tag

                for j in range(self.block_size):
                    self.data[cache_line][j] = self.memory.read(block_address + j)
                self.data[cache_line][offset] = data
                
                self.lru[index * self.associativity:(index + 1) * self.associativity] \
                    += self.lru[index * self.associativity:(index + 1) * self.associativity] < self.lru[cache_line] & \
                    self.valid_bits[index * self.associativity:(index + 1) * self.associativity]
                self.lru[cache_line] = 0
            elif self.replacement_policy == "FIFO":
                cache_line = index * self.associativity + self.fifo[index]
                self.valid_bits[cache_line] = True
                self.tags[cache_line] = tag

                for j in range(self.block_size):
                    self.data[cache_line][j] = self.memory.read(block_address + j)

                self.data[cache_line][offset] = data
                self.fifo[index] = (self.fifo[index] + 1) % self.associativity
            elif self.replacement_policy == "random":
                self._associativity_check[:] = self.valid_bits[index * self.associativity:(index + 1) * self.associativity] & \
                        (self.tags[index * self.associativity:(index + 1) * self.associativity] == tag)
                associative_index = np.argmax(self._associativity_check)
                if self._associativity_check[associative_index]:
                    cache_line = index * self.associativity + associative_index
                    if not self.valid_bits[cache_line]:
                        self.valid_bits[cache_line] = True
                        self.tags[cache_line] = tag

                        for j in range(self.block_size):
                            self.data[cache_line][j] = self.memory.read(block_address + j)

                        self.data[cache_line][offset] = data
                else:
                    cache_line = index * self.associativity + np.random.randint(self.associativity)
                    self.valid_bits[cache_line] = True
                    self.tags[cache_line] = tag

                    for j in range(self.block_size):
                        self.data[cache_line][j] = self.memory.read(block_address + j)

                    self.data[cache_line][offset] = data
                
        self.memory.write(address=address, data=data)

    def cache_reset(self) -> None:
        """
        Resets the cache.
        """
        self.tags.fill(0)
        self.valid_bits.fill(False)
        self.data.fill(0)

        if self.replacement_policy == "LRU":
            self.lru.fill(0)
        
        if self.replacement_policy == "FIFO":
            self.fifo.fill(0)
        
        self.loads = 0
        self.load_hits = 0

        self.writes = 0
        self.write_hits = 0
    
    @property
    def get_loads(self) -> int:
        return self.loads
    
    @property
    def get_load_hits(self) -> int:
        return self.load_hits
    
    @property
    def get_load_misses(self) -> int:
        return self.loads - self.load_hits
    
    @property
    def get_writes(self) -> int:
        return self.writes
    
    @property
    def get_write_hits(self) -> int:
        return self.write_hits
    
    @property
    def get_write_misses(self) -> int:
        return self.writes - self.write_hits
        