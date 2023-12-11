# Audio LED Matrix

This repository describes how to create an nxn LED matrix that lights up to
display a real-time equalizer program as a demonstration of the skills learned
by the Khalifa University ECCE402 class of Fall 2023.

### Hardware implementation

The final objective is to physically implement an LED matrix that lights up
according to the magnitude of the frequency bin it corresponds to.

#### The weapon of choice: Raspberry Pi 4
The following kind of setup will suffice to build a mental picture as the
project goes on.
Something much easier to control than individual LEDs such as this [MAX7219 Dot
Matrix Display Module for the Raspberry Pi
Pico](https://www.instructables.com/Raspberry-Pi-Pico-MAX7219-8x8-Dot-Matrix-Scrolling/)
can be used to display an LED matrix of the output can be used to display an
LED matrix of the output.

### Dependencies
The following Python libraries were used to perform the signal processing,
rendering, and acquisition.

* [Numpy](https://github.com/numpy/numpy)

* [Matplotlib](https://github.com/matplotlib/matplotlib)

* [Scipy](https://github.com/scipy/scipy)

* [Luma LED Matrix Drivers] (https://pypi.org/project/luma.led-matrix)
    * [Full installation guide] (https://luma-led-matrix.readthedocs.io/en/latest/install.html)
* sounddevice
    Depends on libportaudio2
    ```sudo apt-get install libportaudio2```
    On Rasbperry Pi 4.

Next, install all dependencies using
    ```pip3 install -r requirements.txt```
A minimum of Python 3.9 is required.

