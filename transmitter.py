import streamlit as st
import numpy as np
import sounddevice as sd

st.title("Simple Tone Generator: Type to Play a-z + Space")

# -------------------------
# 1) Setup / Configuration
# -------------------------

# Characters: a–z plus a blank space
letters = [chr(i) for i in range(97, 123)] + [" "]  # 27 total

# Split 1000 Hz to 10000 Hz into 27 sections (a-z + space)
NUM_SECTIONS = len(letters)  # 27
FREQ_LOW = 1000.0
FREQ_HIGH = 10000.0
SECTION_WIDTH = (FREQ_HIGH - FREQ_LOW) / NUM_SECTIONS

# Precompute a “center frequency” for each character
base_freqs = []
for i in range(NUM_SECTIONS):
    freq_center = FREQ_LOW + (i + 0.5) * SECTION_WIDTH
    base_freqs.append(freq_center)

# Audio parameters
SAMPLE_RATE = 44100
DURATION = 0.3   # seconds per tone
VOLUME = 0.5     # amplitude scale

# -------------------------
# 2) Helper Functions
# -------------------------

def letter_to_freq(ch: str) -> float:
    """Return the center frequency for the given letter/space."""
    ch = ch.lower()
    if ch in letters:
        idx = letters.index(ch)
        return base_freqs[idx]
    else:
        return 0.0  # Invalid char => no tone

def play_tone(freq: float, duration: float = DURATION, volume: float = VOLUME):
    """Generate and play a sine wave at freq (Hz) for duration (s)."""
    if freq <= 0:
        return
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = volume * np.sin(2 * np.pi * freq * t)
    sd.play(wave, samplerate=SAMPLE_RATE)
    sd.wait()  # Wait until playback finishes

# -------------------------
# 3) Main Streamlit UI
# -------------------------

# Session state to track typed text and last length
if "typed_text" not in st.session_state:
    st.session_state.typed_text = ""
if "last_len" not in st.session_state:
    st.session_state.last_len = 0

typed = st.text_input("Type letters (a–z) or space to hear tones:", st.session_state.typed_text)

# Check if new characters were typed
current_len = len(typed)
if current_len > st.session_state.last_len:
    # Extract only the newly typed portion
    new_chars = typed[st.session_state.last_len:]
    for ch in new_chars:
        if ch.lower() in letters:
            freq = letter_to_freq(ch)
            play_tone(freq)
    st.session_state.typed_text = typed
    st.session_state.last_len = current_len
elif current_len < st.session_state.last_len:
    # User backspaced or shortened the text
    st.session_state.typed_text = typed
    st.session_state.last_len = current_len

st.write("Start typing above to hear tones for each character!") 
