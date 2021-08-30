# pueo-python-math
PUEO python code for planning (filter work, etc.)

Seriously need to add more info here, but the "omg this is crazy help me get started" is:

1. Find a quick primer on Python if you're not familiar
2. Find a quick primer on Scipy and signal filtering if you're not familiar

Then you can look at the pueo.py code. The "init" section sets up filters that we'll eventually need and also reads in the system's impulse response.

Basic flow is

Event generation occurs in "get()":

1. Take the impulse response, and shift it by a random fraction of a sample (since we don't know when the impulse comes in)
2. Generate Gaussian (white, flat frequency) noise
3. Bandpass filter the Gaussian noise to our passband.
4. Scale the impulse response to the desired SNR and add to noise.
5. Generate desired CW interference, if any, and add to signal.

Then you can look in pueo_class_test for some of the trigger testing, however that notebook is a *disaster*.
