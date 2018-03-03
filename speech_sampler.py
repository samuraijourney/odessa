import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import fractions
import time
import ctypes
import multiprocessing
from scipy import signal
from matplotlib import animation
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.pools import ProcessPool, ThreadPool

class Speech_Sampler():

    def __init__(self, window_duration = 2):
        self.__window_duration = window_duration # seconds
        self.__fs = 16000 # time resolution of the recording device (Hz)

        length = self.__window_duration * self.__fs

        self.__data_update_interval = 0.025 # seconds
        self.__signal_plot_refresh_interval = 0.025 # seconds
        self.__spectrogram_plot_refresh_interval = 0.5 # seconds
        self.__energy_plot_refresh_interval = 0.025 # seconds
        self.__zero_crossings_plot_refresh_interval = 0.025 # seconds
        self.__pause_stop_draw_duration = 0.5 # seconds
        self.__silence_samples = int(0.01 * self.__fs)
        self.__data = np.zeros(length)
        self.__force_draw = False
        self.__hide_signal_plot = False
        self.__hide_spectrogram_plot = False
        self.__hide_energy_plot = False
        self.__hide_zero_crossing_plot = False
        self.__last_pause_state = False
        self.__last_updated_spectrogram_time = 0
        self.__last_updated_signal_time = 0
        self.__last_updated_energy_time = 0
        self.__last_updated_zero_crossings_time = 0
        self.__pause = self.__last_pause_state
        self.__silence_energy = np.zeros(length)
        self.__silence_energy_max_thresholds = np.zeros(length)
        self.__silence_energy_min_thresholds = np.zeros(length)
        self.__silence_zero_crossings = np.zeros(length)
        self.__silence_speech_detect = np.zeros(length)
        self.__silence_zero_crossing_threshold = 0
        self.__silence_sample_count = 0
        self.__silence_threshold_samples = 100
        self.__silence_threshold_samples_speech_detection = 50 * self.__silence_samples
        self.__silence_std_deviation = 0
        self.__tracking_index = int(length - self.__silence_samples / 2 - 1)

    def __audio_callback(self, indata, frames, time, status):
        data = indata[:, 0]
        shift = len(data)

        self.__data = np.roll(self.__data, -shift, axis = 0)
        self.__data[-shift:] = data[:]
        self.__tracking_index = self.__tracking_index - shift

        self.__silence_energy = np.roll(self.__silence_energy, -shift, axis = 0)
        self.__silence_energy[-shift:] = 0
        self.__silence_energy_max_thresholds = np.roll(self.__silence_energy_max_thresholds, -shift, axis = 0)
        self.__silence_energy_max_thresholds[-shift:] = 0
        self.__silence_energy_min_thresholds = np.roll(self.__silence_energy_min_thresholds, -shift, axis = 0)
        self.__silence_energy_min_thresholds[-shift:] = 0
        self.__silence_zero_crossings = np.roll(self.__silence_zero_crossings, -shift, axis = 0)
        self.__silence_zero_crossings[-shift:] = 0
        self.__silence_speech_detect = np.roll(self.__silence_speech_detect, -shift, axis = 0)
        self.__silence_speech_detect[-shift:] = 0

        signal_samples_radius = self.__silence_samples / 2
        signal_start_index = self.__tracking_index
        signal_end_index = self.__tracking_index + shift

        threshold_samples_radius = self.__silence_threshold_samples_speech_detection / 2
        threshold_start_index = signal_start_index - threshold_samples_radius
        threshold_end_index = signal_end_index - threshold_samples_radius

        signal_samples = self.__build_sample_matrix(self.__data, signal_start_index, signal_end_index, signal_samples_radius * 2)

        # Compute energy
        self.__silence_energy[signal_start_index : signal_end_index] = self.__calculate_energy(signal_samples)
        
        # Compute energy thresholds
        energy_samples = self.__build_sample_matrix(self.__silence_energy, threshold_start_index, threshold_end_index, threshold_samples_radius * 2)
        energy_thresholds = self.__calculate_energy_threshold(energy_samples)
        self.__silence_energy_min_thresholds[threshold_start_index : threshold_end_index] = energy_thresholds[0, :]
        self.__silence_energy_max_thresholds[threshold_start_index : threshold_end_index] = energy_thresholds[1, :]

        # Compute zero crossings    
        self.__silence_zero_crossings[signal_start_index : signal_end_index] = self.__calculate_zero_crossings(signal_samples)

        # Compute zero-crossing thresholds
        if (self.__silence_sample_count < self.__silence_threshold_samples) and ((self.__silence_sample_count + shift) >= self.__silence_threshold_samples):
            zero_crossings_samples = self.__silence_zero_crossings[threshold_start_index : threshold_end_index]
            self.__silence_zero_crossing_threshold = self.__calculate_zero_crossing_threshold(zero_crossings_samples)

        self.__tracking_index = self.__tracking_index + shift
        self.__silence_sample_count = self.__silence_sample_count + shift

    def __build_sample_matrix(self, data, start_index, end_index, samples_per_column):
        matrix = np.zeros((samples_per_column, end_index - start_index))
        radius = samples_per_column / 2
        for i in range(start_index, end_index):
            matrix[:, i - start_index] = data[i - radius : i + radius]
        return matrix

    def __calculate_energy(self, data):
        return np.sum(np.abs(data), axis = 0)

    def __calculate_energy_threshold(self, energies):
        max_energy = np.max(energies, axis = 0)
        min_energy = np.min(energies, axis = 0)

        min_threshold_energy = np.min((0.03 * (max_energy - min_energy) + min_energy, 4 * min_energy), axis = 0)
        max_threshold_energy = 5 * min_threshold_energy

        return np.stack((min_threshold_energy, max_threshold_energy), axis = 0)

    def __calculate_zero_crossings(self, data):
        zero_crossings = 0

        data_roll = np.roll(data, 1, axis = 0)
        zero_crossings = np.sum(np.abs(np.subtract(np.sign(data[1:, :]), np.sign(data_roll[1:, :]))), axis = 0) / (2 * len(data))
        
        return zero_crossings

    def __calculate_zero_crossing_threshold(self, zero_crossings):
        return min(25, np.mean(zero_crossings) + 2 * np.std(zero_crossings))

    def __create_plots(self):
        plot_count = 0
        plot_index = 0

        if (not self.__hide_signal_plot):
            plot_count = plot_count + 1
        if (not self.__hide_energy_plot):
            plot_count = plot_count + 1
        if (not self.__hide_zero_crossing_plot):
            plot_count = plot_count + 1
        if (not self.__hide_spectrogram_plot):
            plot_count = plot_count + 1

        self.__fig, axes = plt.subplots(plot_count, 1)

        # Initialize all plots
        if (not self.__hide_signal_plot):
            self.__initialize_signal_plot(axes[plot_index], self.__data[:], self.__silence_speech_detect[:])
            plot_index = plot_index + 1
        if (not self.__hide_energy_plot):
            self.__initialize_energy_plot(axes[plot_index], self.__silence_energy[:], self.__silence_energy_min_thresholds[:], self.__silence_energy_max_thresholds[:])
            plot_index = plot_index + 1
        if (not self.__hide_zero_crossing_plot):
            self.__initialize_zero_crossings_plot(axes[plot_index], self.__silence_zero_crossings[:])
            plot_index = plot_index + 1
        if (not self.__hide_spectrogram_plot):
            self.__initialize_spectrogram_plot(axes[plot_index], self.__data[:])
            axes[plot_index].axis('off')

        self.__fig.tight_layout(pad = 0)

    def __find_speech_start(self, energies, zero_crossings):
        min_energy_crossed = False
        max_energy_crossed = False

        energy_index = np.nan
        min_energy_first = False

        min_energy_threshold, max_energy_threshold = self.__calculate_energy_threshold(energies)

        # Search for energy start
        for i in range(1, len(energies)):
            min_crossing = np.sign(energies[-i - 1] - min_energy_threshold) + np.sign(energies[-i] - min_energy_threshold)
            max_crossing = np.sign(energies[-i - 1] - max_energy_threshold) + np.sign(energies[-i] - max_energy_threshold)

            # Crossing max twice
            if (max_crossing == 0) and max_energy_crossed:
                return np.nan

            # Crossing min twice
            if (min_crossing == 0) and min_energy_crossed:
                return np.nan

            # Crossing min once
            if min_crossing == 0:
                min_energy_crossed = True

            # Crossing max once
            if max_crossing == 0:
                max_energy_crossed = True

            # Crossing neither in the first iteration
            if (not min_energy_crossed) and (not max_energy_crossed):
                return np.nan
            elif i == 1:
                if min_energy_crossed:
                    min_energy_first = True

            # Crossed min and max
            if min_energy_crossed and max_energy_crossed:
                if min_energy_first:
                    energy_index = -1
                else:
                    energy_index = -i
                break

        if np.isnan(energy_index):
            return np.nan

        # Search for zero crossings        
        if np.abs(energy_index - 25) > (len(zero_crossings) + 1):
            return np.nan

        start_index = np.nan
        above_zero_crossing_threshold_count = 0

        for i in range(0, 25):
            above_zero_crossing_threshold = max(np.sign(zero_crossings[energy_index - i] - self.__silence_zero_crossing_threshold), 0)
            above_zero_crossing_threshold_count = above_zero_crossing_threshold_count + above_zero_crossing_threshold

            if above_zero_crossing_threshold == 1:
                start_index = energy_index - i
        
        if above_zero_crossing_threshold_count < 3:
            return np.nan
        
        return start_index

    def __find_speech_start2(self, energies, zero_crossings, energy_min_thresholds, energy_max_thresholds, zero_crossing_thresholds):
        n1 = np.nan
        n2 = np.nan

        # Find first instance where we dip below the ITL from the end
        for i in range(1, len(energies)):
            index = -i - 1
            if energies[i] > energy_min_thresholds[i]:
                continue
            
        for i in range(-index - 1, len(energies)):
            index = -i - 1
            if energies[i] < energy_min_thresholds[i]:
                continue
        
        #if index + 25 

        for i in range(0, 25):
            index = -i - 1
            above_zero_crossing_threshold = max(np.sign(zero_crossings[index] - zero_crossing_thresholds[index]), 0)
            #above_zero_crossing_threshold_count = above_zero_crossing_threshold_count + above_zero_crossing_threshold

            #if above_zero_crossing_threshold == 1:
           #     start_index = energy_index - i
        
       #if above_zero_crossing_threshold_count < 3:
        #    return np.nan
        
        #return start_index

    def __initialize_energy_plot(self, ax, data, min_threshold_data, max_threshold_data):
        self.__energy_plot = ax
        self.__energy_plot_data = ax.plot(data)
        self.__energy_plot_min_data = ax.plot(min_threshold_data, color='r')
        self.__energy_plot_max_data = ax.plot(max_threshold_data, color='r')

        ax.axis((0, len(data), 0, 1))
        ax.set_title("Energy")
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='on', labelleft='on')

    def __initialize_signal_plot(self, ax, signal_data, speech_detect_data):
        time = np.linspace(-self.__window_duration, 0, len(signal_data))
        
        self.__signal_plot = ax
        self.__signal_plot_data = ax.plot(time, signal_data)
        self.__signal_plot_speech_detect_data = ax.plot(time, speech_detect_data, color='r')
        self.__garbage_plot_data = ax.plot(time, np.full(len(signal_data), np.nan))

        ax.set_title("Audio")
        ax.axis((0, len(signal_data), -0.25, 0.25))
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        ax.set_xlabel("Time (s)")
        ax.set_xlim([-self.__window_duration, 0])
        ax.tick_params(bottom='on', top='off', labelbottom='on', right='off', left='on', labelleft='on')

    def __initialize_spectrogram_plot(self, ax, data):
        self.__spectrogram_plot = ax

        ax.set_title("Spectrogram")
        ax.specgram(data, NFFT = 1024, Fs = self.__fs, noverlap = 900)

    def __initialize_zero_crossings_plot(self, ax, data):
        self.__silence_zero_crossings_plot = ax
        self.__silence_zero_crossings_plot_data = ax.plot(data)
        self.__silence_zero_crossings_plot_threshold_data = ax.plot(np.full(len(data), self.__silence_zero_crossing_threshold), color='r')

        ax.axis((0, len(data), 0, 0.1))
        ax.set_title("Zero Crossings")
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='on', labelleft='on')

    def __scale_plot(self, ax, data, grow_only = True):
        max_val = max(data)
        min_val = min(data)
        padding = (max_val - min_val) * 0.1

        if padding == 0.0:
            padding = 0.05

        if grow_only == True:
            min_val = min(ax.get_ylim()[0], min_val - padding)
            max_val = max(ax.get_ylim()[1], max_val + padding)
        else:
            min_val = min_val - padding
            max_val = max_val + padding

        ax.set_ylim(bottom = min_val, top = max_val)

    def __update_energy_plot(self, data, min_threshold_data, max_threshold_data):
        self.__scale_plot(self.__energy_plot, data, False)
        self.__scale_plot(self.__energy_plot, min_threshold_data, True)
        self.__scale_plot(self.__energy_plot, max_threshold_data, True)

        for _,line in enumerate(self.__energy_plot_data):
            line.set_ydata(data)
        for _,line in enumerate(self.__energy_plot_min_data):
            line.set_ydata(min_threshold_data)
        for _,line in enumerate(self.__energy_plot_max_data):
            line.set_ydata(max_threshold_data)

    def __update_plots(self, frame):
        if self.__force_draw == True:
            self.__animation.event_source.stop()

        if ((not self.__hide_signal_plot) and (((time.time() - self.__last_updated_signal_time) > self.__signal_plot_refresh_interval) or self.__force_draw)):
            self.__update_signal_plot(self.__data[:], self.__silence_speech_detect[:])
            self.__last_updated_signal_time = time.time()

        if ((not self.__hide_energy_plot) and (((time.time() - self.__last_updated_energy_time) > self.__energy_plot_refresh_interval) or self.__force_draw)):
            self.__update_energy_plot(self.__silence_energy[:], self.__silence_energy_min_thresholds[:], self.__silence_energy_max_thresholds[:])
            self.__last_updated_energy_time = time.time()

        if ((not self.__hide_zero_crossing_plot) and (((time.time() - self.__last_updated_zero_crossings_time) > self.__zero_crossings_plot_refresh_interval) or self.__force_draw)):
            self.__update_zero_crossings_plot(self.__silence_zero_crossings[:])
            self.__last_updated_zero_crossings_time = time.time()

        if ((not self.__hide_spectrogram_plot) and (((time.time() - self.__last_updated_spectrogram_time) > self.__spectrogram_plot_refresh_interval) or self.__force_draw)):
            self.__update_spectrogram_plot(self.__data[:])
            self.__last_updated_spectrogram_time = time.time()

        if (not self.__pause) or self.__force_draw:
            plt.pause(0.01)

        if (self.__force_draw == True):
            self.__force_draw = False

        if (self.__last_pause_state == False) and (self.__pause == True):
            self.__force_draw = True

        self.__last_pause_state = self.__pause

        return self.__garbage_plot_data

    def __update_signal_plot(self, signal_data, detect_data):
        self.__scale_plot(self.__signal_plot, signal_data, False)
        for _,line in enumerate(self.__signal_plot_data):
            line.set_ydata(signal_data)
        for _,line in enumerate(self.__signal_plot_speech_detect_data):
            line.set_ydata(detect_data)

    def __update_spectrogram_plot(self, data):
        self.__spectrogram_plot.clear()
        self.__spectrogram_plot.specgram(data, NFFT = 1024, Fs = self.__fs, noverlap = 900)
        self.__spectrogram_plot.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='on', labelleft='on')
        self.__spectrogram_plot.set_title("Spectrogram")

    def __update_zero_crossings_plot(self, data):
        threshold_data = np.full(len(data), self.__silence_zero_crossing_threshold)

        self.__scale_plot(self.__silence_zero_crossings_plot, data, False)
        self.__scale_plot(self.__silence_zero_crossings_plot, threshold_data, True)

        for _,line in enumerate(self.__silence_zero_crossings_plot_data):
            line.set_ydata(data)
        for _,line in enumerate(self.__silence_zero_crossings_plot_threshold_data):
            line.set_ydata(threshold_data)

    def hide_energy_plot(self, hide = True):
        self.__hide_energy_plot = hide

    def hide_signal_plot(self, hide = True):
        self.__hide_signal_plot = hide

    def hide_spectrogram_plot(self, hide = True):
        self.__hide_spectrogram_plot = hide

    def hide_zero_crossing_plot(self, hide = True):
        self.__hide_zero_crossing_plot = hide

    def pause(self):
        self.__pause = True

    def resume(self):
        self.__pause = False
        self.__animation.event_source.start()

    def start(self):
        self.__create_plots()
        
        # Need reference to animation otherwise garbage collector removes it...
        self.__animation = animation.FuncAnimation(self.__fig, self.__update_plots, interval = self.__data_update_interval * 1000, blit = True)
        stream = sd.InputStream(channels=1, samplerate=self.__fs, callback=self.__audio_callback)

        with stream:
            plt.show(block = False)
            print("")
            while True:
                raw_input("Press any key to pause")
                self.pause()
                raw_input("Press any key to resume")
                self.resume()

if __name__ == '__main__':
    sampler = Speech_Sampler(5)
    sampler.hide_spectrogram_plot()
    #sampler.hide_energy_plot()
    #sampler.hide_zero_crossing_plot()
    sampler.start()