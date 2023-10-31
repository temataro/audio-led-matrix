#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pydub
import sys
import os.path


def read_audio(audio_file):
    a = pydub.AudioSegment.from_mp3(audio_file)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))[0]  # Spectrum data contrained
    # in one channel is plenty.
    return a.frame_rate, np.float32(y) / 2**15  # normalize from 0 to 1


def main():
    audio_file = "never.mp3"
    if not os.path.exists(audio_file):
        print("File not found!")
        return 0

    sr, a = read_audio(audio_file)
    print(f"sample rate: {sr}, array length: {len(a)}")

    t = np.arange(len(a)) / sr
    plt.plot(t, a)
    plt.xlabel("Time in seconds")
    plt.ylabel("Signal Amplitude")
    plt.title(f"Channel 1 of {audio_file}")
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
