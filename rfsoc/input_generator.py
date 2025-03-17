import numpy as np
from utils import floating_point_to_fixed_point

class InputGenerator:
    
    def __init__(self, FFT_SIZE):
        self.FFT_SIZE = FFT_SIZE
        self.TOTAL_SAMPLES = self.FFT_SIZE

        self.FRAC_BITS = 15
        self.TOTAL_BITS = 16
        self.max_amplitude = 0.9
    
    def generate_sinewave(self):
        
        input_signal_uint = np.empty(self.TOTAL_SAMPLES, dtype=np.uint32)

        # Parameters
        data_size = self.FFT_SIZE  # Number of samples
        sampling_rate = data_size  # Sampling rate (matches the number of samples for 1 cycle)
        max_amplitude = 0.9  # Max amplitude of the sine wave


        # Generate the sine wave
        t = np.arange(data_size)  # Time indices

        sine_wave = np.empty(self.TOTAL_SAMPLES, dtype=np.float16)
        sine_wave = max_amplitude * np.sin(2 * np.pi * 2 * t / self.FFT_SIZE)
        real_input_signal_fixed = [floating_point_to_fixed_point(np.real(val), self.TOTAL_BITS, self.FRAC_BITS) for val in sine_wave]

        for i in range(self.FFT_SIZE):
            real_part = real_input_signal_fixed[i] & 0xFFFF
            imag_part = 0x0000

            # Pack real and imaginary into a 32-bit word (real: upper 16 bits, imag: lower 16 bits)
            input_signal_uint[i] = (real_part << 16) | imag_part
                
                
        return sine_wave, input_signal_uint 

    def packing(self, wav_data):
        num_samples = len(wav_data)
        input_signal_uint = np.empty(num_samples, dtype=np.uint32)
        
        for i in range(num_samples):
            real_part = wav_data[i] & 0xFFFF
            imag_part = 0x0000

            # Pack real and imaginary into a 32-bit word (real: upper 16 bits, imag: lower 16 bits)
            input_signal_uint[i] = (real_part << 16) | imag_part

        return input_signal_uint