import numpy as np

@staticmethod
def floating_point_to_fixed_point(value, total_bits, fractional_bits):
    # Scale the value
    scale_factor = 2**fractional_bits
    fp_value = int(np.round(value * scale_factor))
    
    # Define range
    max_value = 2**(total_bits - 1) - 1
    min_value = -2**(total_bits - 1)
    
    # Clamp to range
    if fp_value > max_value:
        print(f"Overflow occurred: {value} was rounded down to {max_value / scale_factor}")
        fp_value = max_value
    elif fp_value < min_value:
        print(f"Underflow occurred: {value} was rounded up to {min_value / scale_factor}")
        fp_value = min_value

    # Return the clamped fixed-point value
    return fp_value

@staticmethod
def fixed_point_to_floating_point(fixed_value, total_bits, fractional_bits):
    # Convert to signed integer directly (Python handles two's complement)
    if fixed_value >= 2**(total_bits - 1):
        fixed_value -= 2**total_bits  # Handle negative values
    
    # Scale back to floating-point
    return fixed_value / (2**fractional_bits)


@staticmethod
def calculateSQNR(quantized_fft_signal, ideal_fft_signal):
    assert np.shape(quantized_fft_signal) == np.shape(ideal_fft_signal), "The two FFT results have different shape."
    
    signal_power = np.sum(np.abs(quantized_fft_signal) ** 2)
    noise_power = np.sum(np.abs(quantized_fft_signal - ideal_fft_signal) ** 2)
    sqnr = 10 * np.log10(signal_power / noise_power)
    return sqnr


@staticmethod
def unpackOutput(output_buffer, FFT_POINTS):
    # Unpack the output buffer into real and imaginary parts
    out_real = np.zeros(FFT_POINTS, dtype=np.float16)
    out_imag = np.zeros(FFT_POINTS, dtype=np.float16)
    
    fractional_bits = 15

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
        
    return out_real + 1j * out_imag