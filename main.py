
import numpy as np
import matplotlib.pyplot as plt

from model import Model

"""
Dependencies:
numpy 
matplotlib
numba (speeds simulation up: pre-compiles functions to machine code)
"""


def main():
    # tiny example program:
    cells, eod_freqs, parameters = load_model_csv("models.csv")

    example_cell_idx = 20

    print("Example with cell: {}".format(cells[example_cell_idx]))
    model = Model(parameters[example_cell_idx])
    eod_frequency = eod_freqs[example_cell_idx]

    step = model.parameters["step_size"]
    stimulus_length = 2  # in seconds

    time = np.arange(0, stimulus_length, step)

    # create EOD-like stimulus with an amplitude step
    stimulus = np.sin(2*np.pi*eod_frequency*time)
    # include an amplitude step after 0.5s until 1.5s in the stimulus with given contrast
    contrast = 1.3
    stimulus[10000:30000] *= contrast

    spikes = model.simulate(stimulus)

    spikes = np.array(spikes)

    freq = calculate_isi_frequency(spikes, step)
    freq_time = np.arange(spikes[0], spikes[-1], step)

    fig, axes = plt.subplots(2, 1, sharex="col")

    axes[0].plot(time, stimulus)
    axes[0].set_title("Stimulus")
    axes[0].set_ylabel("Amplitude in mV")

    axes[1].plot(freq_time, freq)
    axes[1].set_title("Model Frequency")
    axes[1].set_ylabel("Frequency in Hz")
    axes[1].set_xlabel("Time in s")
    plt.show()
    plt.close()


def load_model_csv(file):
    cells = []
    eod_freqs = []
    parameters = []
    with open(file, 'r') as file:
        header_line = file.readline()
        header_parts = header_line.strip().split(",")
        keys = header_parts[2:]

        for line in file:
            line_parts = line.strip().split(",")
            cells.append(line_parts[0])
            eod_freqs.append(float(line_parts[1]))
            parameter = {}
            for i in range(len(keys)):
                parameter[keys[i]] = float(line_parts[i+2])

            parameters.append(parameter)

    return cells, eod_freqs, parameters


def calculate_isi_frequency(spikes, step):
    """
    calculates inter-spike interval frequency
    (wasn't tested a lot may give different length than time = np.arange(spikes[0], spikes[-1], step),
    or raise an index error for some inputs)

    :param spikes: spike time points
    :param step: step size of the model

    :return: the frequency trace:
                starts at the time of first spike
                ends at the time of the last spike.
    """

    isis = np.diff(spikes)
    freq_points = 1 / isis
    freq = np.zeros(int((spikes[-1] - spikes[0]) / step))

    current_idx = 0
    for i, isi in enumerate(isis):
        end_idx = int(current_idx + np.rint(isi / step))
        freq[current_idx:end_idx] = freq_points[i]
        current_idx = end_idx

    return freq


if __name__ == '__main__':
    main()
