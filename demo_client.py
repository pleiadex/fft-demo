import socket
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import unpackOutput

# Configuration parameters
HOST = "pynq"        # Replace with your board's IP address
PORT = 5001          # Must match the server port
FFT_POINTS = 1024    # Length of FFT vector from the board
fractional_bits = 15 # Fixed-point fractional bits for conversion
SAMPLE_RATE = 32000  # Assumed sample rate in Hz (adjust as needed)

# Total bytes per message: 2 vectors * 1024 elements * 2 bytes/element = 4096 bytes
expected_bytes = 2 * FFT_POINTS * 2

# Create and connect the client socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
print(f"Connected to server at {HOST}:{PORT}")

# Set up the Matplotlib plot
# Frequency axis for the positive frequencies (only half of FFT points are unique for real signals)
freqs = np.linspace(0, SAMPLE_RATE / 2, FFT_POINTS // 2)
fig, ax = plt.subplots()
line, = ax.plot(freqs, np.zeros_like(freqs))
ax.set_xlim(0, SAMPLE_RATE / 2)
# ax.set_ylim(0, 100)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Scaled Amplitude")
ax.set_title("Real-time FFT Visualization from Z1 Board")

def update(frame):
    global client_socket

    start_time = time.time()
    # Receive a complete block of FFT data from the socket
    data = b""
    while len(data) < expected_bytes:
        packet = client_socket.recv(expected_bytes - len(data))
        if not packet:
            print("Socket closed by the server.")
            client_socket.close()
            plt.close(fig)
            return line,
        data += packet

    print(f"Time to receive data: {(time.time() - start_time) * 1000:.3f} ms")

    start_time = time.time()

    # Convert the received bytes into a NumPy array of uint32 and reshape it into 1024 elements
    output_buffer = np.frombuffer(data, dtype=np.int32).reshape((FFT_POINTS,))

    # Reconstruct the complex FFT output vector
    rfsoc_fft_out = unpackOutput(output_buffer, FFT_POINTS)

    print(f"Time to unpack data: {(time.time() - start_time) * 1000:.3f} ms")


    start_time = time.time()
    # Use only the first half (unique frequencies) for plotting
    magnitude = np.abs(rfsoc_fft_out[:FFT_POINTS // 2])

    # Scale the magnitude so that the maximum value is 100 (for visualization purposes)
    max_mag = np.max(magnitude)
    if max_mag > 0:
        scaled_magnitude = (magnitude / max_mag) * 100
    else:
        scaled_magnitude = magnitude

    # Update the plot with the new data
    line.set_ydata(magnitude)
    print(f"Time to update plot: {(time.time() - start_time) * 1000:.3f} ms")
    return line,

# Set up the animation: update the plot every 10 ms
ani = FuncAnimation(fig, update, interval=10, blit=True)

try:
    plt.show()
except KeyboardInterrupt:
    print("Client interrupted and stopping...")
finally:
    client_socket.close()
    print("Connection closed.")
