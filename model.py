
import numpy as np
from numba import jit


class Model:
    DEFAULT_VALUES = {"mem_tau": 0.015,
                      "v_base": 0.,
                      "v_zero": 0.,
                      "threshold": 1,
                      "v_offset": -10.,
                      "input_scaling": 60.,
                      "delta_a": 0.08,
                      "tau_a": 0.1,
                      "a_zero": 2.,
                      "noise_strength": 0.05,
                      "step_size": 0.00005,
                      "dend_tau": 0.001,
                      "refractory_period": 0.001}

    def __init__(self, parameters=None):
        if parameters is None:
            self.parameters = self.DEFAULT_VALUES
        else:
            self.parameters = parameters

        self.v1 = np.array([])
        self.adaption = np.array([])
        self.spiketimes = []
        self.input_voltage = np.array([])

    def simulate(self, stimulus):
        v_zero = self.parameters["v_zero"]
        a_zero = self.parameters["a_zero"]
        step_size = self.parameters["step_size"]
        threshold = self.parameters["threshold"]
        v_base = self.parameters["v_base"]
        delta_a = self.parameters["delta_a"]
        tau_a = self.parameters["tau_a"]
        v_offset = self.parameters["v_offset"]
        mem_tau = self.parameters["mem_tau"]
        noise_strength = self.parameters["noise_strength"]
        input_scaling = self.parameters["input_scaling"]
        dend_tau = self.parameters["dend_tau"]
        ref_period = self.parameters["refractory_period"]

        parameters = np.array([v_zero, a_zero, step_size, threshold, v_base, delta_a, tau_a, v_offset, mem_tau, noise_strength,
                      input_scaling, dend_tau, ref_period])

        output_voltage, adaption, spiketimes, input_voltage = simulate_fast(stimulus, *parameters)
        self.v1 = output_voltage
        self.adaption = adaption
        self.spiketimes = spiketimes
        self.input_voltage = input_voltage

        return spiketimes


@jit(nopython=True)
def simulate_fast(stimulus_array, v_zero, a_zero, step_size, threshold, v_base, delta_a, tau_a, v_offset, mem_tau, noise_strength, input_scaling, dend_tau, ref_period):

    # rectify stimulus array:
    stimulus_array[stimulus_array<0.0] = 0.0


    length = len(stimulus_array)
    output_voltage = np.zeros(length)
    adaption = np.zeros(length)
    input_voltage = np.zeros(length)

    spiketimes = []
    output_voltage[0] = v_zero
    adaption[0] = a_zero
    input_voltage[0] = stimulus_array[0]

    for i in range(1, length, 1):

        noise_value = np.random.normal()
        noise = noise_strength * noise_value / np.sqrt(step_size)

        input_voltage[i] = input_voltage[i - 1] + (
                    (-input_voltage[i - 1] + stimulus_array[i]) / dend_tau) * step_size

        output_voltage[i] = output_voltage[i - 1] + ((v_base - output_voltage[i - 1] + v_offset + (
                    input_voltage[i] * input_scaling) - adaption[i - 1] + noise) / mem_tau) * step_size

        adaption[i] = adaption[i - 1] + ((-adaption[i - 1]) / tau_a) * step_size

        # refractory period
        if len(spiketimes) > 0 and (step_size * i) - spiketimes[-1] < ref_period + step_size/2:
            output_voltage[i] = v_base

        # spiking
        if output_voltage[i] > threshold:
            output_voltage[i] = v_base
            spiketimes.append(i * step_size)
            adaption[i] += delta_a / tau_a

    return output_voltage, adaption, spiketimes, input_voltage