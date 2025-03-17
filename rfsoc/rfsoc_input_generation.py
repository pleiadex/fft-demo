import time
import numpy as np
from input_generator import InputGenerator


# Constants
FFT_SIZE = 2 ** 16       # 65536 samples per chunk

fractional_bits = 15     # Fractional bits for fixed-point scaling
total_bits = 16          # Total bits in the fixed-point format

# Generate input signal
ig = InputGenerator(FFT_SIZE)
print("Input waveform generation starts")

start_time = time.time()  # Record the start time
sine_wave, input_signal_uint = ig.generate_sinewave()  # Generate the waveform
elapsed_time = time.time() - start_time  # Calculate the elapsed time

print(f"Input waveform is ready and took {elapsed_time:.3f}s") 


# Save the array
np.save("cpu_input_signal_64K.npy", sine_wave)
np.save("rfsoc_input_signal_64K.npy", input_signal_uint)


