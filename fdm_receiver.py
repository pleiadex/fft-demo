import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyaudio
import struct
import time
import threading
import streamlit as st

# --- Initialize Streamlit session state variables if needed ---
if 'decoded_buffer' not in st.session_state:
    st.session_state['decoded_buffer'] = ""
if 'fft_started' not in st.session_state:
    st.session_state['fft_started'] = False
if 'matplotlib_thread' not in st.session_state:
    st.session_state['matplotlib_thread'] = None

# --- Utility Function to Unpack FFT Output ---
def unpackOutput(buffer):
    """
    Unpack a 32-bit packed FFT output into a complex numpy array.
    Upper 16 bits are the real part, lower 16 bits are the imaginary part.
    Both parts are assumed to be in signed 16-bit two's complement.
    """
    # Extract real part (upper 16 bits) and interpret as signed 16-bit integer
    real = (buffer >> 16).astype(np.int16)
    # Extract imaginary part (lower 16 bits) and interpret as signed 16-bit integer
    imag = (buffer & 0xFFFF).astype(np.int16)
    return real + 1j * imag

# --- Only initialize the FFT/acquisition code once ---
if not st.session_state['fft_started']:
    st.session_state['fft_started'] = True

    # --- Configuration Parameters ---
    HOST = "pynq"         # Replace with your Z1 board's IP address
    PORT = 5001           # Must match the server port
    FFT_POINTS = 1024     # Number of FFT points
    SAMPLE_RATE = 48000   # Microphone sample rate
    FRAME_SIZE = 1024     # Number of audio samples per frame
    expected_bytes = FFT_POINTS * 4  # 4 bytes per FFT point

    # --- Create and Connect the Client Socket ---
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    print(f"Connected to server at {HOST}:{PORT}")

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

    def update(frame):
        t0 = time.time()

        # === Stage 1: Audio Sampling (PyAudio) ===
        data = stream.read(FRAME_SIZE, exception_on_overflow=False)
        audio_data = np.array(struct.unpack(str(FRAME_SIZE) + 'h', data), dtype=np.int16)
        print(f"Time taken to sample audio: {(time.time() - t0) * 1000:.3f} ms")

        t0 = time.time()
        # === Stage 2: Data Packing ===
        # Pack 16-bit samples into 32-bit integers: upper 16 bits = sample, lower 16 bits = 0
        packed_data = (audio_data.astype(np.uint16).astype(np.uint32)) << 16
        print(f"Time taken to pack data: {(time.time() - t0) * 1000:.3f} ms")

        t0 = time.time()
        # Send the packed audio data to the server (Z1 board)
        client_socket.sendall(packed_data.tobytes())
        print(f"Time taken to send data: {(time.time() - t0) * 1000:.3f} ms")

        # === Stage 3: Receive FFT Data from Server ===
        data_bytes = b""
        while len(data_bytes) < expected_bytes:
            packet = client_socket.recv(expected_bytes - len(data_bytes))
            if not packet:
                print("Socket closed by the server.")
                client_socket.close()
                plt.close(fig)
                return line,
            data_bytes += packet

        t0 = time.time()
        output_buffer = np.frombuffer(data_bytes, dtype=np.int32).reshape((FFT_POINTS,))
        rfsoc_fft_out = unpackOutput(output_buffer)
        print(f"Time taken to unpack data: {(time.time() - t0) * 1000:.3f} ms")

        t0 = time.time()
        # === Stage 4: Process and Visualize FFT Data ===
        magnitude = np.abs(rfsoc_fft_out[:FFT_POINTS // 2])
        max_mag = np.max(magnitude)
        if max_mag > 0:
            scaled_magnitude = (magnitude / max_mag) * 100
        else:
            scaled_magnitude = magnitude

        line.set_ydata(scaled_magnitude)
        print(f"Time taken to visualize data: {(time.time() - t0) * 1000:.3f} ms")

        # === Stage 5: Frequency-domain Decoding ===
        # Define the frequency range to decode (1 kHz - 10 kHz) split into 27 sections.
        threshold = 80
        decoded_char = None
        low_freq = 1000
        high_freq = 10000
        section_edges = np.linspace(low_freq, high_freq, 28)  # 27 sections -> 28 edges
        # Map sections: indices 0-25 -> 'a' to 'z', index 26 -> blank (space)
        symbols = [chr(i) for i in range(97, 123)] + [' ']
        max_val = 0
        max_section = None
        for i in range(27):
            section_mask = (freqs >= section_edges[i]) & (freqs < section_edges[i+1])
            if np.any(section_mask):
                section_max = np.max(scaled_magnitude[section_mask])
                if section_max > threshold and section_max > max_val:
                    max_val = section_max
                    max_section = i
        if max_section is not None:
            decoded_char = symbols[max_section]
            # Append the decoded character to the global buffer
            st.session_state.decoded_buffer += decoded_char
            print(f"Decoded char: {decoded_char}")

        return line,

    ani = FuncAnimation(fig, update, interval=10, blit=True)

    # --- Define a cleanup function for when the app is stopped ---
    def cleanup():
        stream.stop_stream()
        stream.close()
        p.terminate()
        client_socket.close()
        print("Connection closed.")

    import atexit
    atexit.register(cleanup)

    # --- Function to run the matplotlib FFT plot in a separate thread ---
    def run_matplotlib():
        plt.show()

    # Start the matplotlib window in a background thread if not already running.
    if st.session_state['matplotlib_thread'] is None:
        st.session_state['matplotlib_thread'] = threading.Thread(target=run_matplotlib, daemon=True)
        st.session_state['matplotlib_thread'].start()


# --- Streamlit App UI ---
st.title("Decoded Characters")
st.write("This area shows the characters decoded from the FFT data (frequency–domain modulation).")

if st.button("Clear"):
    st.session_state.decoded_buffer = ""

st.text_area("Decoded Buffer", st.session_state.decoded_buffer, height=200)

# --- Auto–refresh the Streamlit app to show the latest decoded text ---
try:
    # st_autorefresh is available in recent versions of Streamlit (or via the streamlit_autorefresh package)
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=500, key="datarefresh")
except ImportError:
    st.write("Auto–refresh not available. Please refresh the page to see updates.")
