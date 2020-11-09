"""Generating input and target"""

import numpy as np
import torch.utils.data as data


class StaticInput(data.Dataset):
    def __init__(
            self,
            time_length,
            time_scale,
            value_min,
            value_max,
            signal_length,
            variable_signal_length,
            sigma_in,
            delay_variable):
        self.time_length = time_length
        self.time_scale = time_scale
        self.value_min = value_min
        self.value_max = value_max
        self.signal_length = signal_length
        self.variable_signal_length = variable_signal_length
        self.sigma_in = sigma_in
        self.delay_variable = delay_variable

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        # input signal
        signal = np.zeros(self.time_length + 1)
        signal_timing = 0
        vs = np.random.randint(-self.variable_signal_length, self.variable_signal_length + 1)
        signal_length = self.signal_length + vs
        value = np.random.rand() * (self.value_max - self.value_min) + self.value_min
        t = np.arange(0, signal_length * self.time_scale, self.time_scale)
        if len(t) != signal_length:
            t = t[:-1]
        signal_ = value * np.ones(t)
        signal[signal_timing: signal_timing + signal_length] = signal_
        signal += np.random.normal(0, self.sigma_in, self.time_length)

        # target
        target = np.array([value])

        signal = np.expand_dims(signal, axis=1)

        return signal, target
