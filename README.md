# Audio LED Matrix

This repository describes how to create an nxn LED matrix that lights up to
display a real-time equalizer program as a demonstration of the skills learned
by the Khalifa University ECCE402 class of Fall 2023.

## Background Theory

### DTFTs, DFTs, FFTs and FFT Buckets

### Hardware implementation

The final objective is to physically implement an LED matrix that lights up
according to the magnitude of the frequency bin it corresponds to.

#### The weapon of choice: RP2040-Zero
The following kind of setup will suffice to build a mental picture as the
project goes on.
[WokWi Arduino Simulator MicroPython 6x6 LED matrix Control](https://wokwi.com/projects/379957049597714433)
Or, alternatively, something much easier to control such as this [MAX7219 Dot Matrix Display Module for the Raspberry Pi Pico](https://www.instructables.com/Raspberry-Pi-Pico-MAX7219-8x8-Dot-Matrix-Scrolling/)

### Dependencies
The following Python libraries were used to perform the signal processing,
rendering, and acquisition.

* [Numpy](https://github.com/numpy/numpy)

* [Matplotlib](https://github.com/matplotlib/matplotlib)

* [Pydub](https://github.com/jiaaro/pydub)

            pip install numpy matplotlib pydub


(Pydub is dependent on ffmpeg:    ``` sudo apt install ffmpeg ```)
