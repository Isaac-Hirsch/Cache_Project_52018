"""
This file implements the CPU class, which is responsible for executing instructions.
"""

import numpy as np
from memory import Memory, Cache

DOUBLE_SIZE = 8

class CPU:
    def __init__(self, memory: Memory, cache: Cache):
        self.memory = memory
        self.cache = cache

        self.registers = np.zeros(3, dtype=np.float64)

        self.instruction_count = 0

        self._double_bytes_array = np.zeros(DOUBLE_SIZE, dtype=np.ubyte)
        self._double = np.zeros(1, dtype=np.float64)
    
    def loadDouble(self, address: int) -> np.float64:
        """
        Load a double from memory.
        
        Args:
            address (int): The address to load from.
            register (int): The register to load into.
        Returns:
            np.float64: The value loaded from memory.
        """

        self.instruction_count += 1

        for i in range(DOUBLE_SIZE):
            block, offset = self.cache.read(address + i)
            self._double_bytes_array[i] = block[offset]

        return self._double_bytes_array.view(np.float64)[0]
    
    def storeDouble(self, address: int, value: np.float64) -> None:
        """
        Store a double in memory.

        Args:
            address (int): The address to store at.
            value (np.float64): The value to store.
        """

        self.instruction_count += 1
        self._double[0] = value
        value_bytes = self._double.view(np.ubyte)
        for i in range(DOUBLE_SIZE):
            self.cache.write(address + i, value_bytes[i])

    def addDouble(self, value1: np.float64, value2: np.float64) -> np.float64:
        """
        Add two doubles.

        Args:
            value1 (np.float64): The first value.
            value2 (np.float64): The second value.
        Returns:
            np.float64: The result of the addition.
        """

        self.instruction_count += 1
        return value1 + value2

    def multDoubles(self, value1: np.float64, value2: np.float64) -> np.float64:
        """
        Multiply two doubles.

        Args:
            value1 (np.float64): The first value.
            value2 (np.float64): The second value.
        Returns:
            np.float64: The result of the multiplication.
        """

        self.instruction_count += 1
        return value1 * value2
    
    def loadAddDouble(self, address1: int, address2: int, address3: int) -> np.float64:
        """
        Loads two doubles, adds them, and stores the result in a third address.
        This method is uses 4 instructions.

        Args:
            address1 (int): The address of the first double.
            address2 (int): The address of the second double.
            address3 (int): The address to store the result.
        
        Returns:
            np.float64: The result of the addition.
        """

        self.registers[0] = self.loadDouble(address1)
        self.registers[1] = self.loadDouble(address2)
        self.registers[2] = self.addDouble(self.registers[0], self.registers[1])
        self.storeDouble(address3, self.registers[2])

        return self.registers[2]
    
    def loadMultDouble(self, address1: int, address2: int, address3: int) -> np.float64:
        """
        Loads two doubles, multiplies them, and stores the result in a third address.
        This method is uses 4 instructions.

        Args:
            address1 (int): The address of the first double.
            address2 (int): The address of the second double.
            address3 (int): The address to store the result.
        
        Returns:
            np.float64: The result of the multiplication.
        """

        self.registers[0] = self.loadDouble(address1)
        self.registers[1] = self.loadDouble(address2)
        self.registers[2] = self.multDoubles(self.registers[0], self.registers[1])
        self.storeDouble(address3, self.registers[2])

        return self.registers[2]
    
    def loadFMADouble(self, address1: int, address2: int, address3: int, address4: int) -> np.float64:
        """
        Loads two doubles, multiplies them, adds the result to a third double, and stores the result in a fourth address.
        This method is uses 6 instructions.

        Args:
            address1 (int): The address of the first double to multiply.
            address2 (int): The address of the second double to multiply.
            address3 (int): The address of the double to add.
            address4 (int): The address to store the result.
        
        Returns:
            np.float64: The result of the FMA operation.
        """

        self.registers[0] = self.loadDouble(address1)
        self.registers[1] = self.loadDouble(address2)
        self.registers[2] = self.multDoubles(self.registers[0], self.registers[1])
        self.registers[0] = self.loadDouble(address3)
        self.registers[1] = self.addDouble(self.registers[0], self.registers[2])
        self.storeDouble(address4, self.registers[1])

        return self.registers[1]
    
    @property
    def getInstructionCount(self) -> int:
        """
        Get the number of instructions executed.
        
        Returns:
            int: The number of instructions executed.
        """

        return self.instruction_count
