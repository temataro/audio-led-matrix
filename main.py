#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pydub
import sys
import os.path


def read_audio(audio_file):
    a = pydub.AudioSegment.from_file(audio_file)
    y = np.array(a.get_array_of_samples())
    return a.frame_rate, np.float32(y) / 2**15  # normalize from 0 to 1


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

    # TODO: Fit all the freq bins into a specific bucket
    total_energy = np.sum(np.square(np.abs(A)))  # sum of energy in signal
    # SUM[ ||x(k)||^2 ]
    bucket_energies = [
        np.sum(np.square(np.abs(A[i * N // buckets : (i + 1) * N // buckets])))
        for i in range(buckets)
    ]
    print("Here is the energy per bucket: ")
    print(*bucket_energies, sep="\n")
    print(
        f"CHECK. Total energy: {total_energy:0.4f}, sum of energy in buckets: {np.sum(bucket_energies)}"
    )
    new_bins = np.arange(sr / -2, sr / 2, sr / buckets)
    plt.plot(
        new_bins, 20 * np.log10(bucket_energies), "."
    )  # TODO: Replace this with a bar graph
    plt.show()

    return freq, 20 * np.log10(np.abs(A))


def main():
    audio_file = "never.mp3"
    if not os.path.exists(audio_file):
        print("File not found!")
        return 0

    sr, a = read_audio(audio_file)
    print(f"sample rate: {sr}, array length: {len(a)}")

    t = np.arange(len(a)) / sr
    # Plot signal in time
    plt.plot(t, a)
    plt.xlabel("Time in seconds")
    plt.ylabel("Signal Amplitude")
    plt.title(f"Channel 1 of {audio_file}")
    plt.show()

    # Plot FFT
    freq, A = make_fft_buckets(a, 10, sr)
    plt.plot(
        freq[len(freq) // 2 :], A[len(freq) // 2 :]
    )  # only plotting positive frequencies
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
