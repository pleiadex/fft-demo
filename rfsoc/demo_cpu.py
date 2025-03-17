import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile

# FFT parameters
FFT_SIZE = 2**10  # 64K points

# Load the WAV file (signed 16-bit)
sample_rate, data = wavfile.read("recorded_1024.wav")

# If stereo, convert to mono by averaging channels
if data.ndim > 1:
    data = data.mean(axis=1)

# Ensure data is in signed 16-bit format
data = data.astype(np.int16)

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
    segment = data[current_index:current_index + FFT_SIZE]
    current_index += FFT_SIZE

    # Compute the FFT and take only the positive frequencies (first half)
    fft_result = np.fft.fft(segment, n=FFT_SIZE)
    magnitude = np.abs(fft_result[:FFT_SIZE // 2])
    
    # Scale the magnitude so the maximum value is 100
    max_mag = np.max(magnitude)
    if max_mag > 0:
        scaled_magnitude = (magnitude / max_mag) * 100
    else:
        scaled_magnitude = magnitude

    # Update the plot
    line.set_ydata(scaled_magnitude)
    return line,

# Use FuncAnimation to update the FFT plot repeatedly
ani = FuncAnimation(fig, update, interval=10, blit=True)
plt.show()
