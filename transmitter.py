import streamlit as st
import numpy as np
import sounddevice as sd
import time

st.title("Tone Generator (Transmitter)")

# ------------------------------------------------------------
# 1) CONFIGURATION / SETUP
# ------------------------------------------------------------

letters = [chr(i) for i in range(97, 123)] + [" "]  # 26 letters + blank
NUM_SECTIONS = len(letters)  # 27
FREQ_LOW = 1000.0
FREQ_HIGH = 10000.0
SECTION_WIDTH = (FREQ_HIGH - FREQ_LOW) / NUM_SECTIONS

base_freqs = []
for i in range(NUM_SECTIONS):
    freq_center = FREQ_LOW + (i + 0.5) * SECTION_WIDTH
    base_freqs.append(freq_center)

DEFAULT_FS = 44100  # sample rate
DEFAULT_DURATION = 0.1  # Short duration for each beep
DEFAULT_VOLUME = 0.5  # amplitude scale
GAP_DURATION = 0.1  # Gap between tones

# ------------------------------------------------------------
# 2) HELPER FUNCTIONS
# ------------------------------------------------------------

def letter_to_freq(letter: str) -> float:
    letter = letter.lower()
    if letter in letters:
        idx = letters.index(letter)
        return base_freqs[idx]
    return 0.0

def play_tone(freq: float, duration: float = DEFAULT_DURATION, volume: float = DEFAULT_VOLUME):
    if freq <= 0:
        return
    t = np.linspace(0, duration, int(DEFAULT_FS * duration), endpoint=False)
    wave = volume * np.sin(2 * np.pi * freq * t)
    # Apply fade-in (first 20%) and fade-out (last 20%) to reduce clicks
    fade_samples = int(0.2 * len(wave))
    envelope = np.ones(len(wave))
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)  # Fade in
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)  # Fade out
    wave = wave * envelope
    sd.play(wave, samplerate=DEFAULT_FS)
    sd.wait()

def play_word(word: str, freq_offset: float = 0.0):
    # Play a short silent "warm-up" tone to stabilize audio system
    silent_warmup = np.zeros(int(DEFAULT_FS * 0.01))  # 10ms silence
    sd.play(silent_warmup, samplerate=DEFAULT_FS)
    sd.wait()
    
    # Play each tone with a gap
    for ch in word:
        freq = letter_to_freq(ch) + freq_offset
        play_tone(freq, duration=DEFAULT_DURATION)
        time.sleep(GAP_DURATION)  # Gap between letters

# ------------------------------------------------------------
# 3) STREAMLIT UI: TEXT INPUT AND TRANSMIT BUTTON
# ------------------------------------------------------------

st.subheader("Transmit Your Sentence")
sentence = st.text_input("Enter a sentence (a-z and spaces only):", "")
if st.button("Transmit"):
    if sentence:
        play_word(sentence, freq_offset=0.0)
        st.success(f"Transmitted: {sentence}")
    else:
        st.warning("Please enter a sentence to transmit.")

# ------------------------------------------------------------
# 4) KEYBOARD LAYOUT
# ------------------------------------------------------------

st.subheader("Virtual Keyboard (a–z + space)")

if "last_letter" not in st.session_state:
    st.session_state.last_letter = None

ROW_SIZE = 7
for row_start in range(0, len(letters), ROW_SIZE):
    row_letters = letters[row_start: row_start + ROW_SIZE]
    cols = st.columns(len(row_letters))
    for i, letter in enumerate(row_letters):
        btn_label = f"**{letter}**" if letter == st.session_state.last_letter else letter
        if cols[i].button(btn_label):
            st.session_state.last_letter = letter
            freq = letter_to_freq(letter)
            play_tone(freq)

# ------------------------------------------------------------
# 5) PREDEFINED WORD BUTTONS
# ------------------------------------------------------------

st.subheader("Transmit Predefined Words")

word_list = ["Hello", "Streamlit", "Pynq", "Test", "Demo"]
for w in word_list:
    if st.button(w):
        play_word(w, freq_offset=0.0)
        st.success(f"Transmitted: {w}")