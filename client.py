import socket
import numpy as np

# Replace with your PYNQ-Z1 board's IP address
HOST = "pynq"  # e.g., "192.168.1.100"
PORT = 5001           # Same port as used by the server

FFT_POINTS = 2**10
fractional_bits = 15

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
print(f"Connected to server at {HOST}:{PORT}")

# Total bytes per message: 2 vectors * 1024 elements * 2 bytes/element
expected_bytes = 2 * 1024 * 2 # 4KB

try:
    while True:
        # Receive the fixed-size block
        data = b""
        while len(data) < expected_bytes:
            packet = client_socket.recv(expected_bytes - len(data))
            if not packet:
                break
            data += packet

        if len(data) < expected_bytes:
            print("Incomplete data received. Closing connection.")
            break

        # Convert the bytes back to a NumPy array and reshape
        output_buffer = np.frombuffer(data, dtype=np.uint32).reshape((1, 1024))
        
        # Unpack the output buffer into real and imaginary parts
        out_real = np.zeros(FFT_POINTS, dtype=np.float32)
        out_imag = np.zeros(FFT_POINTS, dtype=np.float32)

        for i in range(FFT_POINTS):
            packed = output_buffer[i]
            
            # Extract real and imaginary parts (16-bit values)
            real_value = (packed >> 16) & 0xFFFF  # Real part is in the upper 16 bits
            imag_value = packed & 0xFFFF          # Imaginary part is in the lower 16 bits
            
            
            #Handle sign extension for 16-bit fixed-point values
            if real_value & 0x8000:  # Check if the sign bit is set (negative number)
                real_value -= 0x10000  # Apply sign extension for negative values

            if imag_value & 0x8000:  # Check if the sign bit is set (negative number)
                imag_value -= 0x10000  # Apply sign extension for negative values

            # Convert from fixed-point (fi16_15) to floating-point
            out_real[i] = real_value / (2**fractional_bits)
            out_imag[i] = imag_value / (2**fractional_bits)
        
        rfsoc_fft_out = (out_real + 1j * out_imag)

    

except KeyboardInterrupt:
    print("Client interrupted and stopping...")

finally:
    client_socket.close()
    print("Connection closed.")