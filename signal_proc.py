import numpy as np
from scipy import signal
from sci_plot import *

t = np.linspace(0, 1, 1000)
freq = 5

signal = np.sin(2*np.pi*freq*t)
noise = np.random.normal(0, 0.05, signal.shape)
noisy_signal = signal + noise

plot_sequence(sequences=[noisy_signal],
              labels=["Signal"],
              xlabel=r"$t$",
              ylabel="Amplitude",)
