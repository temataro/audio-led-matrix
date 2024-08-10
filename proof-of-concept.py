#!/usr/bin/python3

"""
The idea is to simply demonstrate how an equalizer would look if you stepped through
each frame of the LED matrix output manually (press q in matplotlib to close a figure).
Convince our TA to let us do this.

Acknowledgements:
    This team was comprised of (all!) 5 students in the course ECCE402 Digital Signals Processing
    given in Fall 2023 under Dr. Paschalis Sofotasios: Amira Alshamsi, Hassan Aboelseoud, Hassan Safa,
    Olyad Emiru, and Temesgen Ataro.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import os.path


audio_file = "never.wav"


def read_audio(audio_file):
    sr, arr = wavfile.read(audio_file)
    if arr.shape[0] == 2:  # for 2 channel audio sources, only use one channel
        arr = np.array(arr[1])
    return sr, arr


def make_fft_buckets(audio, buckets, sr):
    """
    Take an audio array and perform an FFT. Then spread the contents of
    sr/len(audio) frequency bins into a pre-determined number of larger
    buckets.
    returns a histogram of FFT values with their energies.
    """
    # TODO: Demonstrate Parseval's Theorem by finding the energy in the
    # time domain signal and the total energy stored in the frequency buckets.

    A = np.fft.fftshift(np.fft.fft(audio))
    N = len(A)
    freq = np.arange(sr / -2, sr / 2, sr / len(audio))  # our FFT frequency bins

    total_energy = np.sum(np.square(np.abs(A)))  # sum of energy in signal
    # SUM[ ||x(k)||^2 ]
    bucket_energies = [
        np.sum(np.square(np.abs(A[i * N // buckets : (i + 1) * N // buckets])))
        for i in range(buckets)
    ]
    # print("Here is the energy per bucket: ")
    # print(*bucket_energies, sep="\n")
    # print(
    #     f"CHECK. Total energy: {total_energy:0.4f}, sum of energy in buckets: {np.sum(bucket_energies)}"
    # )
    new_bins = [f"{x * sr // buckets} Hz" for x in range(buckets)]

    return new_bins, 20 * np.log10(bucket_energies)


def segment_audio(arr, ms_segments, pos, sr=44_100, windowing="none"):
    """
    Takes an array and yields a specific slice of it to be processed.
    # of samples = ms_segments * sr / 1000
    """
    seg_size = int(ms_segments * sr / 1000)
    segments = int(arr.size // seg_size)
    # print(
    #     f"Total_audio_length {arr.size / sr} seconds. Segment_Length {seg_size / sr} seconds. #_of_segments {segments}"
    # )
    # TODO: Implement some sectin overlap or windowing in output
    match windowing:
        case "hamming":
            return arr[pos * seg_size : (pos + 1) * seg_size] * np.hamming(seg_size)
        case "hanning":
            return arr[pos * seg_size : (pos + 1) * seg_size] * np.hanning(seg_size)
        case "bartlett":
            return arr[pos * seg_size : (pos + 1) * seg_size] * np.bartlett(seg_size)
        case "none":
            return arr[pos * seg_size : (pos + 1) * seg_size]


def display_graphs(a, sr):
    ms_segments = 500
    seg_size = int(ms_segments * sr / 1000)
    num_segments = int(a.size // seg_size)
    for i in range(num_segments):
        section = segment_audio(
            a, ms_segments, i, sr, windowing="hamming"
        )  # select a random segment to test
        t = np.arange(len(section)) / sr
        # Plot signal in time
        plt.plot(t, section)
        plt.xlabel("Time in seconds")
        plt.ylabel("Signal Amplitude")
        plt.title(f"Channel 1 of {audio_file}")
        plt.show()

        bins, A = make_fft_buckets(section, 10, sr)
        # Plot FFT
        plt.bar(bins, A)  # only plotting positive frequencies
        plt.show()


def main():
    if not os.path.exists(audio_file):
        print("File not found!")
        return 0

    sr, a = read_audio(audio_file)
    print(f"sample rate: {sr}, array length: {len(a)}")
    display_graphs(a, sr)


if __name__ == "__main__":
    sys.exit(main())
