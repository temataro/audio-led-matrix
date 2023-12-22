#!/usr/bin/python3
# Project libraries
from processing import *

# External libraries (installed and built-in)
from scipy.io import wavfile
import sounddevice as sd
from PIL import Image, ImageDraw, ImageFont

from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
import tkinter as tk
import numpy as np
from itertools import count
import time
import math
import os

WIDTH, HEIGHT = 16, 8  # Matrix configuration (each matrix is 8 rows * 8 col)
NUM_BUCKETS = WIDTH  # Represents the number of frequency buckets in the LED matrix.
HEIGHT_BUCKETS = HEIGHT  # Represents the height of each frequency bucket in the LED matrix.
NUM_KNOBS = WIDTH  # Represents the number of potentiometer knobs available.
BUCKETS_PER_PTM = NUM_BUCKETS // NUM_KNOBS
PATH = "../audio/never.wav"  # Path to the audio file. Must be a .wav file.


def paint_led_matrix(spectrum: np.array, device: max7219) -> None:
    """
    Paint the (one-sided) magnitude spectrum passed as a Numpy array onto a
    WIDTH x HEIGHT LED matrix. "WIDTH" represents the number of frequency
    buckets and "HEIGHT" represents the number of LEDs per bucket available for
    display. The spectrum is expected to be provided for a real array of
    magnitudes from 0 to 1.
    """

    # Downsize the spectrum to fit "WIDTH" buckets.
    step = math.ceil(len(spectrum) / WIDTH)
    spectrum = np.hstack((spectrum, np.zeros(step * WIDTH - len(spectrum))))[
        step // 2 :: step
    ]
    # Scale the spectrum to the height of the LED matrix and round to nearest
    # integer.
    spectrum = (np.rint(HEIGHT * spectrum)).astype(int)

    # Paint the LED matrix
    In = list((2 ** spectrum) - 1)
    N = len(In)
    x1 = In[N // 2 :]
    x2 = In[: N // 2]
    y = [
        x1[i // 2] if i % 2 == 0 else x2[i // 2] for i in range(N)
    ]  # interleave x1 and x2
    y = [y[i] if y[i] > 0 else 0 for i in range(N)]  # ensure data doesn't have negative
    # values before converting to bytes
    image = y[::-1]
    image_data = bytes(image)
    img = Image.frombytes("1", (WIDTH, HEIGHT), image_data)
    device.display(img)

    return spectrum


def read_potentiometers(
    knobs: list[tk.Scale],
    knob_max: float,
    width: int = 4,
    height: int = 2,
    buckets_per_ptm: int = 2,
) -> np.array:
    """
    Read the values input by the potentiometers into a NumPy array representing
    the amplification (EQ) to be applied to each frequency bucket. "width"
    represents the number of potentiometers available, while "buckets_per_ptm"
    represents how many frequency buckets are controlled by each potentiometer.
    "height" represents the maximum amplification that can be applied to a
    particular bucket. The return value is a width x buckets_per_ptm array
    containing the respective amplifications per bucket.
    """

    # Read values from potentiometer knobs normalized from 0 to 1.
    potentiometers = np.array([knob.get() for knob in knobs]) / float(knob_max)

    # Scale the amplification array by the maximum amplification and resize to
    # fit buckets.
    potentiometers = height * potentiometers
    potentiometers = np.repeat(potentiometers, buckets_per_ptm)

    return potentiometers


def main_pi(reverb: float = 0) -> None:
    """
    Main function for the entire input -> processing -> output pipeline for
    equalizing audio from a .wav file. The audio is processed in intervals of
    duration time_interval. "reverb" is an optional parameter to add a reverb
    ratio from 0 to 1.
    """
    time_interval = 0.1  # Represents the duration (in sec) of each chunk to be processed.

    # Initialize LED matrix.
    serial = spi(port=0, device=0, gpio=noop())
    device = max7219(serial, width=HEIGHT, height=WIDTH, rotate=1)

    # Initialize digital equalizer controls.
    root = tk.Tk()
    root.title("Graphic Equalizer (EQ) Controls")
    knob_max = 200
    title_label = tk.Label(root, text="Graphic Equalizer (EQ) Controls")
    knobs = [
        tk.Scale(root, from_=knob_max, to=0, width=20, length=200)
        for i in range(NUM_KNOBS)
    ]
    labels = [tk.Label(root, text=f"Bin {i}") for i in range(NUM_KNOBS)]
    title_label.grid(row=0, column=0, columnspan=8)
    for i, (knob, label) in enumerate(zip(knobs, labels)):
        knob.set(100)
        knob.grid(row=1, column=i, padx=5, pady=5)
        label.grid(row=2, column=i, padx=5, pady=5)

    # Read the .wav audio file as a time series stored in "wave" with sampling
    # rate "sr".
    sr, wave = wavfile.read(PATH)

    # Calculate the prototype band-pass filter response. This part only needs
    # to be calculated once since the filter bank is just shifted copies of the
    # filter. This makes processing much more efficient.
    num_samples = int(time_interval * sr)
    nfft = 2 ** math.ceil(math.log2(num_samples)) + 1
    half_spectrum = (nfft + 1) // 2
    width_filter = math.ceil(half_spectrum / NUM_BUCKETS)
    prototype = band_pass_filter(
        width_filter * NUM_BUCKETS, width_filter, width_filter // 2
    )

    # Play (non-blocking) the .wav file. The time series can be modified as it
    # is played (not thread safe but is handy for our application).
    sd.play(wave)
    num_errors = 0

    # Infinite loop with counter starting at 1
    for i in count(1):
        # Timer to deduct processing time
        t0 = time.perf_counter()

        # Update control window.
        root.update()

        # Window the current chunk. This program skips processing the first
        # chunk.
        start = i * num_samples
        end = start + num_samples

        # Add reverb by averaging with previous interval. Reverb must be
        # between 0 and 1.
        if reverb > 1 or reverb < 0:
            raise Exception("Reverb ratio must be between 0 and 1.")
        chunk_time = (1 - reverb) * wave[start:end] + reverb * wave[
            start - num_samples : start
        ]

        # Apply the EQ to the waveform. Round up the number of samples to the
        # nearest power of 2 then add 1 for the number of FFT points. This is
        # because an even-numbered FFT does not have an equal number of
        # components for positive and negative frequencies. We need to truncate
        # the FFT to a one-sided FFT to apply the filter to the one-sided
        # spectrum. Otherwise, applying the filter to the two-sided spectrum
        # without errors is somewhat difficult to set up.
        eq_input = read_potentiometers(
            knobs, knob_max, width=NUM_BUCKETS, buckets_per_ptm=BUCKETS_PER_PTM
        )
        chunk_freq, _ = compute_spectrum(chunk_time, nfft=nfft)
        new_chunk_freq = apply_eq(chunk_freq, eq_input, prototype)
        new_chunk_time = compute_waveform(new_chunk_freq, num_samples=num_samples)
        wave[start:end] = new_chunk_time

        # Magnitude spectrum of filtered signal
        new_chunk_freq = np.abs(new_chunk_freq)

        # Used for scaling the spectrum for the LED matrix. As a rule of thumb,
        # the spectrum is normalized with respect to mean + standard deviation.
        # The spectrum is definitely not normally distributed, but it is used
        # as a quick fix. Any value above that is clipped.
        avg = np.mean(new_chunk_freq)
        st_dev = np.std(new_chunk_freq)
        norm = avg + st_dev
        new_chunk_freq = np.clip((new_chunk_freq / norm), 0, 1)

        # Paint the LED matrix.
        paint_led_matrix(new_chunk_freq, device)

        # Timer to deduct processing time
        t1 = time.perf_counter()
        offset = t1 - t0

        # Wait for TIME_INTERVAL seconds to play the current audio chunk
        # (accounting for processing time). A variable is used to keep track of
        # how many times the processing goes out of sync (error checking).
        try:
            time.sleep(time_interval - offset)
        except ValueError as e:
            num_errors += 1
            if num_errors > 3:
                time_interval += 0.1  # reduce framerate
                print(f"{num_errors} skips; reducing framerate to {time_interval}.")


def main_pc():
    root = tk.Tk()
    root.title("Graphic Equalizer (EQ) Controls")
    root.geometry("800x640")
    title_label = tk.Label(root, text="Graphic Equalizer (EQ) Controls")

    def show_play():
        label.config(text=f"{clicked.get()} selected for playback")
        file = clicked.get()
        print(f"{file} selected")
        root.destroy()
        time.sleep(0.001)
        play_on_pc(file)

    def show_alter():
        file = clicked.get()
        print(f"{file} selected")
        alter_on_pc(file)

    files = [i for i in os.listdir() if i[-3:] == "wav"]

    clicked = tk.StringVar(root)
    clicked.set(files[0])
    drop = tk.OptionMenu(root, clicked, *files)
    drop.pack()
    click_for_playback = tk.Button(
        root, text="Select file to view periodogram", command=show_play
    ).pack()
    label = tk.Label(root, text=" ")
    label.pack()
    click_for_noising = tk.Button(
        root, text="Select file to alter", command=show_alter
    ).pack()
    label = tk.Label(root, text=" ")
    label.pack()

    root.mainloop()


if __name__ == "__main__":
    main_pi()
