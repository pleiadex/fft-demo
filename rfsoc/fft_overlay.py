## FFT CLASS TEST

import time
import numpy as np
from pynq.lib.dma import DMA
from pynq import allocate

class FFT:
    def __init__(self, ol, point_size):

        self.dma = ol.axi_dma_0
        self.point_size = point_size
        
    def print_input(self):
        print(self.reversed_input_buf)
        
    def transfer(self, input_buffer):
        if not self.dma:
            raise ValueError("DMA object is not initialized.")
        if len(input_buffer) != self.point_size:
            raise ValueError(f"Input buffer size must be {self.point_size}, but got {len(input_buffer)}.")
    
        # Allocate output buffer
        output_buffer = allocate(shape=(self.point_size,), dtype=np.uint32)

        # Start DMA transfers
        
        
        self.dma.recvchannel.transfer(output_buffer)
        start_time = time.time()
        self.dma.sendchannel.transfer(input_buffer)
        
        # Wait for transfers to complete
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # Invalidate the output buffer for cache coherence (if necessary)
        #output_buffer.invalidate()

        # Measure computation time
        self.computation_time = time.time() - start_time
        return output_buffer
    
    
    def printComputationTime(self):
        print(f"Last FFT computation time: {self.computation_time:.6f} seconds")
        print(f"Throughput: {self.point_size / self.computation_time / 1e3} KS/s")    