from scipy import signal
import numpy as np
import math
from scipy.io import wavfile
import sounddevice as sd
from itertools import count
import time
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk


def apply_window(wave: np.array, window: str | None = "rectangular") -> np.array:
    """
    Apply the window indicated by "window" to the signal indicated by "wave".
    """

    n = len(wave)
    match window:
        case "hann":
            win = np.hanning(n)
        case "hamming":
            win = np.hamming(n)
        case "blackman":
            win = np.blackman(n)
        case _:
            # Default to rectangular window in case of anomalous input
            win = np.ones(n)
    return wave * win


def compute_spectrum(wave: np.array, nfft: int, window: str | None = None) -> np.array:
    """Return the one-sided FFT of the signal indicated by "wave" along with
    the number of samples in "wave", with an optional window applicable.
    """

    if window is not None:
        wave = apply_window(wave, window)
    return np.fft.fft(wave, nfft)[: (nfft + 1) // 2], len(wave)


def band_pass_filter(length: int, interval: int, center: int):
    """
    Returns the (one-sided) frequency spectrum of a band-pass filter with
    a length given by "length", a pass-band given by "interval", and a center
    frequency given by the "center"th sample. It is assumed that all inputs are
    integers, that "interval" divides "length", and that "center" is between
    interval / 2 and length - interval / 2.
    The specs are determined via trial and error to see what the Raspberry Pi
    can handle without excessive delay.
    """

    # Determine specs of band-pass filter normalized from 0 to 1 digital frequency.
    bw_ratio = 0.99  # constant to control the proportion of the interval allotted to the passband
    gpass, gstop = 0.01, 60  # passband and stopband attenuation in dB
    bandwidth = bw_ratio * interval / length
    center_norm = center / length
    fp1, fp2 = center_norm - 0.5 * bandwidth, center_norm + 0.5 * bandwidth
    transition_width = (1 - bw_ratio) / 2 * interval / length
    fs1, fs2 = fp1 - transition_width, fp2 + transition_width

    # Determine IIR filter frequency response.
    # Cascade of SOS (second-order systems) is used to avoid accumulated errors.
    order, wn = signal.cheb1ord([fp1, fp2], [fs1, fs2], gpass, gstop)
    sos = signal.cheby1(order, gpass, wn, btype="bandpass", output="sos")
    _, freq_response = signal.sosfreqz(sos, worN=length)

    return freq_response


def apply_eq(spectrum: np.array, filter_eq: np.array, prototype: np.array) -> np.array:
    """
    Apply the EQ amplifications in "filter_eq" to "spectrum" and return the
    filtered spectrum. The function passes the input spectrum to a filter bank
    of band-pass filters whose number equals the length of filter_eq. If the
    number of filters does not divide the length of the spectrum, the last
    frequency bucket is zero-padded and then re-sliced to the original length.
    """

    # Zero-pad the spectrum appropriately
    original_len = len(spectrum)
    len_filter = math.ceil(original_len / len(filter_eq))
    spectrum = np.hstack(
        (spectrum, np.zeros(len_filter * len(filter_eq) - original_len))
    )

    # Apply the filter bank. "Prototype" is a single BPF at the first bucket.
    filter_bank = np.zeros(len(spectrum))
    for i, gain in enumerate(filter_eq):
        filter_bank = filter_bank + gain * np.roll(prototype, i * len_filter)
    return (spectrum * filter_bank)[:original_len]


def compute_waveform(spectrum: np.array, num_samples: int) -> np.array:
    """
    Return the inverse FFT of the one-sided FFT "spectrum" and truncate it to the
    appropriate number of samples "num_samples". "spectrum" must be of odd
    length for predictable results, so an exception is raised if it is not.
    The real part of the inverse FFT is returned to keep the result real.
    """

    if len(spectrum) % 2 == 0:
        raise Exception("Length of one-sided FFT must be odd.")

    # Append the negative frequency portion of the spectrum.
    side_freq = np.conj(spectrum[1:])[::-1]
    spectrum = np.hstack((spectrum, side_freq))
    wave = np.fft.ifft(spectrum)

    # Convert the inverse FFT to real and truncate to num_samples
    return np.real(wave)[:num_samples]


def play_on_pc(PATH):
    FRAME_RATE = 12
    TIME_INTERVAL = (
        1 / FRAME_RATE
    )  # Represents the duration (in sec) of each chunk to be processed

    # Play (non-blocking) the .wav audio file as a time series stored in "wave"
    # with sampling rate "sr". The time series can be modified as it is played
    # (not thread safe but is handy for our application).
    sr, wave = wavfile.read(PATH)
    sd.play(wave)

    # Initialize periodogram animation
    chunk = int(TIME_INTERVAL * sr)
    fig, ax = plt.subplots()
    # take the periodogram of the first audio frame
    first_frame = wave[:chunk]
    f, Pxx = signal.periodogram(first_frame, sr)

    (line,) = ax.plot(f, np.random.randn(int(np.ceil(chunk / 2))))
    print("_____")
    print(f)
    print("_____")
    ax.set_xlim(0, 22_100)
    ax.set_ylim(0, np.max(Pxx) + 25)
    ax.set_title(f"{PATH[2:-4]}")
    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("volume")
    plt.setp(
        ax,
        xticks=[i for i in np.arange(0, sr / 2 + 10, sr / 10)],
        yticks=[i * np.max(Pxx) for i in np.arange(0, 1.5, 0.25)],
    )

    # show the plot
    plt.show(block=True)

    print("stream started")
    # ====
    # Infinite loop with counter starting at 1
    for i in count(1):
        # Timer to deduct processing time
        t0 = time.perf_counter()

        # Window the current chunk. This program skips processing the first
        # chunk.
        num_samples = int(TIME_INTERVAL * sr)
        start = i * num_samples
        end = start + num_samples

        chunk_time = wave[start:end]
        # Apply the EQ to the waveform. Round up the number of samples to the
        # nearest power of 2 then add 1 for the number of FFT points. This is
        # because an even-numbered FFT does not have an equal number of
        # components for positive and negative frequencies. We need to truncate
        # the FFT to a one-sided FFT to apply the filter to the one-sided
        # spectrum. Otherwise, applying the filter to the two-sided spectrum
        # without errors is somewhat difficult to set up.
        nfft = 2 ** math.ceil(math.log2(num_samples)) + 1
        chunk_freq, _ = compute_spectrum(chunk_time, nfft=nfft)
        new_chunk_time = compute_waveform(chunk_freq, num_samples=num_samples)
        wave[start:end] = new_chunk_time

        # Realtime periodogram view on computer
        _, painted_data = signal.periodogram(chunk_time, fs=sr)
        ax.set_ylim(0, np.max(painted_data) * 1.5)

        line.set_ydata(painted_data)
        fig.canvas.draw()
        fig.canvas.flush_events()
        # Timer to deduct processing time
        t1 = time.perf_counter()
        offset = t1 - t0

        # Wait for TIME_INTERVAL seconds to play the current audio chunk
        # (accounting for processing time).
        time.sleep(TIME_INTERVAL - offset)


def add_noise(PATH, f1, f2, noise_level):
    sr, arr = wavfile.read(PATH)
    noise = (
        np.random.uniform(low=np.min(arr), high=np.max(arr), size=len(arr))
        * noise_level
    )
    # BPF the noise
    bw = f2 - f1
    tw = bw * 0.01
    gpass, gstop = 0.01, 60
    fs1, fs2 = f1 - tw, f2 + tw
    order, wn = signal.cheb1ord([f1, f2], [fs1, fs2], gpass, gstop, fs=sr)
    sos = signal.cheby1(order, rp=0.01, Wn=wn, btype="bandpass", output="sos", fs=sr)
    _, f = signal.sosfreqz(sos, worN=len(arr))

    # apply limits to noise
    N = np.fft.fft(noise)
    N *= f
    arr += np.fft.ifft(N).astype(np.int16)

    # Save audio to file
    wavfile.write(f"{PATH[:-4]}_{f1}_{f2}_{noise_level}.wav", sr, arr.astype(np.int16))


def denoise(PATH, f1, f2):
    sr, arr = wavfile.read(PATH)
    # BSF to filter out noise
    bw = f2 - f1
    tw = bw * 0.001
    gpass, gstop = 0.001, 60
    fs1, fs2 = f1 + tw, f2 - tw

    order, wn = signal.cheb1ord([f1, f2], [fs1, fs2], gpass, gstop, fs=sr)
    sos = signal.cheby1(order, rp=0.01, Wn=wn, btype="bandstop", output="sos", fs=sr)
    _, f = signal.sosfreqz(sos, worN=len(arr))
    arr = np.fft.ifft(np.fft.fft(arr) * f)

    wavfile.write(f"notched_{PATH[:-4]}_{f1}_{f2}.wav", sr, arr.astype(np.int16))


def alter_on_pc(PATH):
    root = tk.Tk()
    root.title("Controls for altering wav file")
    root.geometry("300x200")

    def print_input():
        inp1 = int(inputtxt1.get(1.0, "end-1c"))
        inp2 = int(inputtxt2.get(1.0, "end-1c"))
        noise_level = int(inputtxt3.get(1.0, "end-1c"))

        lbl.config(
            text=f"Additive noise bandlimited from {inp1} Hz - {inp2} Hz added to {PATH}."
        )
        add_noise(PATH, inp1, inp2, noise_level)

    label1 = tk.Label(root, text="Input the frequency limits of additive noise")
    label1.pack()
    inputtxt1 = tk.Text(root, height=1, width=4)
    inputtxt2 = tk.Text(root, height=1, width=4)
    inputtxt1.pack(side=tk.TOP)
    inputtxt2.pack(side=tk.TOP)
    label2 = tk.Label(
        root, text="Input the relative magnitude of noise to add to signal."
    )
    label2.pack()
    inputtxt3 = tk.Text(root, height=1, width=4)
    inputtxt3.pack(side=tk.TOP)

    alter_button = tk.Button(root, text="Alter", command=print_input)
    alter_button.pack()
    lbl = tk.Label(root, text="")
    lbl.pack()

    root.mainloop()


if __name__ == "__main__":
    alter_on_pc("never.mp3")
