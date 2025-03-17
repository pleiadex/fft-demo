import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
from pynq import allocate, PL, Overlay

from input_generator import InputGenerator
from fft_overlay import FFT
from utils import calculateSQNR, unpackOutput

import warnings
warnings.filterwarnings("ignore", message="A NumPy version .* is required for this version of SciPy.*")



# FFT parameters
FFT_SIZE = 2**10  # 1K points

fractional_bits = 15     # Fractional bits for fixed-point scaling
total_bits = 16          # Total bits in the fixed-point format

# Reset PL and load the overlay
PL.reset()
wrapper_name = "1k_FFT"
ol = Overlay(f"{os.getcwd()}/{wrapper_name}.bit")

print('The FFT is programmed onto the RFSoC.')

# Load FFT overlay and allocate input buffer
fft_ol = FFT(ol, FFT_SIZE)
input_buffer = allocate(shape=(FFT_SIZE,), dtype=np.uint32)


# # Load the WAV file (signed 16-bit)
sample_rate, _ = wavfile.read("recorded_1024.wav")

# # If stereo, convert to mono by averaging channels
# if data.ndim > 1:
#     data = data.mean(axis=1)

# # Ensure data is in signed 16-bit format
# data = data.astype(np.int16)

# data type packing
data = np.load("rfsoc_input_signal_uint32.npy") 

data = data[len(data)//3:]

print("The waveform is successfully loaded.")


# Create frequency axis from 0 to Nyquist frequency (sample_rate/2)
freqs = np.linspace(0, sample_rate/2, FFT_SIZE // 2)

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot(freqs, np.zeros_like(freqs))
ax.set_xlim(0, sample_rate/2)
ax.set_ylim(0, 100)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Scaled Amplitude")
ax.set_title("Real-time FFT Visualization")

# Variables for iterating through the audio data
num_samples = len(data)
current_index = 0

def update(frame):
    global current_index

    # Wrap around if we reach the end of the file
    if current_index + FFT_SIZE > num_samples:
        current_index = 0

    # Get the current segment of samples
    input_buffer[:] = data[current_index:current_index + FFT_SIZE]
    current_index += FFT_SIZE

    # Compute the FFT and take only the positive frequencies (first half)
    output_buffer = fft_ol.transfer(input_buffer)  # Execute FFT
    rfsoc_fft_out = unpackOutput(output_buffer, FFT_SIZE)
    magnitude = np.abs(rfsoc_fft_out[:FFT_SIZE // 2])
    
    # Scale the magnitude so the maximum value is 100
    max_val = np.max(magnitude)
    scaled_magnitude = (magnitude / max_val * 100) if max_val > 0 else magnitude

    # Update the plot
    line.set_ydata(scaled_magnitude)
    return line,

# Use FuncAnimation to update the FFT plot repeatedly
ani = FuncAnimation(fig, update, interval=10, blit=True)
plt.show()
