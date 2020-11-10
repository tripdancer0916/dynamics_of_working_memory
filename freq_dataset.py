"""Generating input and target"""

import numpy as np
import torch.utils.data as data


class FreqInput(data.Dataset):
    def __init__(
            self,
            time_length,
            time_scale,
            omega_min,
            omega_max,
            signal_length,
            variable_signal_length,
            sigma_in):
        self.time_length = time_length
        self.time_scale = time_scale
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.signal_length = signal_length
        self.variable_signal_length = variable_signal_length
        self.sigma_in = sigma_in

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        # input signal
        signal = np.zeros(self.time_length + 1)
        signal_timing = 0
        vs = np.random.randint(-self.variable_signal_length, self.variable_signal_length + 1)
        signal_length = self.signal_length + vs
        signal_freq = np.random.rand() * (self.omega_max - self.omega_min) + self.omega_min
        t = np.arange(0, signal_length * self.time_scale, self.time_scale)
        if len(t) != signal_length:
            t = t[:-1]
        # phase_shift = np.random.rand() * np.pi
        # signal_ = np.sin(signal_freq * t + phase_shift)
        signal_ = np.sin(signal_freq * t)
        signal[signal_timing: signal_timing + signal_length] = signal_
        signal += np.random.normal(0, self.sigma_in, self.time_length + 1)

        # target
        target = np.array([signal_freq])

        signal = np.expand_dims(signal, axis=1)

        return signal, target
