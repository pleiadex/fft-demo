import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyaudio
import struct
import time
import sys
import threading

# --- Utility Function to Unpack FFT Output ---
def unpackOutput(buffer):
    """
    Unpack a 32-bit packed FFT output into a complex numpy array.
    Upper 16 bits are the real part, lower 16 bits are the imaginary part.
    Both parts are assumed to be in signed 16-bit two's complement.
    """
    real = (buffer >> 16).astype(np.int16)
    imag = (buffer & 0xFFFF).astype(np.int16)
    return real + 1j * imag

# --- Configuration Parameters ---
HOST = "pynq"         # Replace with your Z1 board's IP address
PORT = 5001           # Must match the server port
FFT_POINTS = 1024     # Number of FFT points
SAMPLE_RATE = 48000   # Microphone sample rate
FRAME_SIZE = 1024     # Number of audio samples per frame
expected_bytes = FFT_POINTS * 4  # Each FFT sample is 4 bytes

# --- Create and Connect the Client Socket ---
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
print(f"Connected to server at {HOST}:{PORT}")
print("Press 'c' to clear the buffer.")

# --- Set Up PyAudio for Microphone Input ---
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=FRAME_SIZE)

# --- Set Up the Matplotlib Plot ---
freqs = np.linspace(0, SAMPLE_RATE / 2, FFT_POINTS // 2)
fig, ax = plt.subplots()
line, = ax.plot(freqs, np.zeros_like(freqs))
ax.set_xlim(0, SAMPLE_RATE / 2)
ax.set_ylim(0, 100)  # Adjust based on your FFT scaling
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Scaled Amplitude")
ax.set_title("Real-time FFT Visualization from Z1 Board")

# Global variables for the decoded sentence and debouncing state.
decoded_buffer = ""
last_decoded_letter = None
last_detected_magnitude = 0  # Unscaled magnitude from last detection

# Define the frequency range to decode (1 kHz - 10 kHz) split into 27 sections.

# FIXME: Adjust the threshold, low_freq, high_freq, and section_edges based on the environment
threshold = 80
unscaled_threshold = 1000
low_freq = 1000
high_freq = 10000
section_edges = np.linspace(low_freq, high_freq, 28)  # 27 sections -> 28 edges

# --- Keyboard Listener for Clearing the Buffer ---
def keyboard_listener():
    global decoded_buffer
    try:
        # Windows: use msvcrt for non-blocking key detection.
        import msvcrt
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                # If user presses "c" or "C", clear the buffer.
                if key.lower() == b'c':
                    decoded_buffer = ""
                    sys.stdout.write("\r" + " " * 80)  # Clear line visually.
                    # sys.stdout.write("\rCleared buffer")
                    sys.stdout.flush()
            time.sleep(0.05)
    except ImportError:
        # On non-Windows systems, consider alternatives like select or curses.
        while True:
            time.sleep(0.1)

# Start the keyboard listener in a separate daemon thread.
keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
keyboard_thread.start()

def update(frame):
    global decoded_buffer, last_decoded_letter, last_detected_magnitude

    # === Stage 1: Audio Sampling (PyAudio) ===
    data = stream.read(FRAME_SIZE, exception_on_overflow=False)
    audio_data = np.array(struct.unpack(str(FRAME_SIZE) + 'h', data), dtype=np.int16)

    # === Stage 2: Data Packing ===
    packed_data = (audio_data.astype(np.uint16).astype(np.uint32)) << 16

    # === Stage 3: Send Data and Receive FFT Data from Server ===
    client_socket.sendall(packed_data.tobytes())
    data_bytes = b""
    while len(data_bytes) < expected_bytes:
        packet = client_socket.recv(expected_bytes - len(data_bytes))
        if not packet:
            print("\nSocket closed by the server.")
            client_socket.close()
            plt.close(fig)
            return line,
        data_bytes += packet

    # === Stage 4: Process and Visualize FFT Data ===
    output_buffer = np.frombuffer(data_bytes, dtype=np.int32).reshape((FFT_POINTS,))
    rfsoc_fft_out = unpackOutput(output_buffer)
    magnitude = np.abs(rfsoc_fft_out[:FFT_POINTS // 2])
    
    # FIXME: Plot just FFT output wihtout scaling
    max_mag = np.max(magnitude)
    if max_mag > 0:
        scaled_magnitude = (magnitude / max_mag) * 100
    else:
        scaled_magnitude = magnitude

    line.set_ydata(scaled_magnitude)

    # === Stage 5: Frequency-domain Decoding with Magnitude-based Debounce ===
    symbols = [chr(i) for i in range(97, 123)] + [' ']
    max_val = 0
    max_section = None
    current_unscaled_magnitude = 0  # Max unscaled magnitude for the detected section

    for i in range(27):
        section_mask = (freqs >= section_edges[i]) & (freqs < section_edges[i+1])
        if np.any(section_mask):
            section_max = np.max(scaled_magnitude[section_mask])
            section_max_unscaled = np.max(magnitude[section_mask])
            if section_max > threshold and section_max_unscaled > unscaled_threshold and section_max > max_val:
                max_val = section_max
                max_section = i
                current_unscaled_magnitude = section_max_unscaled

    if max_section is not None:
        current_letter = symbols[max_section]
        # Debounce based on unscaled magnitude:
        if (last_decoded_letter is None or 
            current_letter != last_decoded_letter or 
            last_detected_magnitude < unscaled_threshold):
            decoded_buffer += current_letter
            last_decoded_letter = current_letter
            last_detected_magnitude = current_unscaled_magnitude
    else:
        last_decoded_letter = None
        last_detected_magnitude = 0

    # --- Terminal Output: Overwrite the current line ---
    sys.stdout.write("\r" + decoded_buffer)  # Extra spaces to clear residual characters.
    sys.stdout.flush()

    return line,

# --- Set Up the Animation (Update every 10 ms) ---
ani = FuncAnimation(fig, update, interval=10, blit=True, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    print("\nClient interrupted and stopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    client_socket.close()
    print("\nConnection closed.")
