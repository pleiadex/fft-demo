import os
import time
import numpy as np

from pynq import allocate, PL, Overlay
import psutil
from tqdm import tqdm  # Progress bar for Python scripts

from fft_overlay import FFT

# ------------------------- Setup & Initialization -------------------------

# Reset PL and load the overlay
print("\n🔄 Resetting Programmable Logic (PL)...")
PL.reset()

wrapper_name = "64k_FFT"
print(f"🔌 Loading Overlay: {wrapper_name}.bit")
ol = Overlay(f"{os.getcwd()}/{wrapper_name}.bit")
print("✅ Overlay successfully loaded!\n")

# Display available memory
mem = psutil.virtual_memory()
print(f"💾 Available Memory: {mem.available / (1024 ** 2):.2f} MB\n")

# ------------------------- Constants & Data Preparation -------------------------

FFT_SIZE = 2 ** 16       # 64K samples per FFT
NUM_ITER = 500          # Total chunks to process
fractional_bits = 15     # Fractional bits for fixed-point scaling
total_bits = 16          # Total bits in the fixed-point format

print("📂 Loading input signal for CPU FFT...")
cpu_input_signal = np.load('cpu_input_signal_64K.npy')
print("✅ CPU input signal loaded!\n")

# ------------------------- CPU FFT Benchmark -------------------------

print("🚀 Starting CPU-based FFT computation...\n")
cpu_start = time.time()

for i in tqdm(range(NUM_ITER), desc="🧮 CPU FFT Progress", unit="iter"):
    np.fft.fft(cpu_input_signal)

cpu_end = time.time()
cpu_duration = cpu_end - cpu_start
print(f"\n✅ CPU FFT Completed! 🕒 Total Execution Time: {cpu_duration:.3f} seconds\n")

# ------------------------- RFSoC FFT Benchmark -------------------------

print("📂 Loading input signal for RFSoC FFT...")
input_buffer = allocate(shape=(FFT_SIZE,), dtype=np.uint32)
input_buffer[:] = np.load('rfsoc_input_signal_64K.npy')
print("✅ RFSoC input signal loaded!\n")

print("🚀 Starting RFSoC-based FFT computation...\n")
fft_ol = FFT(ol, FFT_SIZE)
rfsoc_start = time.time()

for i in tqdm(range(NUM_ITER), desc="⚡ RFSoC FFT Progress", unit="iter"):
    fft_ol.transfer(input_buffer)

rfsoc_end = time.time()
rfsoc_duration = rfsoc_end - rfsoc_start
print(f"\n✅ RFSoC FFT Completed! 🕒 Total Execution Time: {rfsoc_duration:.3f} seconds\n")

# ------------------------- Summary -------------------------

print("📊 Performance Summary:")
print("-" * 35)
print(f"🧮 CPU ({NUM_ITER} x 64K-point FFTs):   {cpu_duration:.3f} seconds")
print(f"⚡ RFSoC ({NUM_ITER} x 64K-point FFTs): {rfsoc_duration:.3f} seconds")
print("-" * 35)
print("🎉 Live demo completed successfully!\n")
