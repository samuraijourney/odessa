from matplotlib import animation
from Queue import Queue
from scipy import signal
from threading import Thread
import fractions
import matplotlib.pyplot as plt
import numpy as np
import os
import sounddevice as sd
import soundfile as sf
import time

class Speech_Sampler():

    def __init__(self, window_duration = 2, fs = 16000):

        # Arguments
        self.__window_duration = window_duration # seconds
        self.__fs = fs # hz

        length = self.__window_duration * self.__fs

        # Plotting
        self.__data_update_interval = 0.025 # seconds
        self.__energy_plot_refresh_interval = 0.025 # seconds
        self.__force_draw = False
        self.__hide_energy_plot = False
        self.__hide_signal_plot = False
        self.__hide_spectrogram_plot = False
        self.__hide_zero_crossing_plot = False
        self.__last_pause_state = False
        self.__last_updated_energy_time = 0
        self.__last_updated_signal_time = 0
        self.__last_updated_spectrogram_time = 0
        self.__last_updated_zero_crossings_time = 0
        self.__pause = self.__last_pause_state
        self.__signal_plot_refresh_interval = 0.025 # seconds
        self.__spectrogram_plot_refresh_interval = 0.5 # seconds
        self.__time = np.linspace(-self.__window_duration, 0, length)
        self.__zero_crossings_plot_refresh_interval = 0.025 # seconds

        # Audio
        self.__audio_frame_size = int(0.01 * self.__fs)
        self.__audio_frame_count = 0
        self.__data = np.zeros(length)
        self.__energies = np.zeros(length)
        self.__energies_max_thresholds = np.zeros(length)
        self.__energies_min_thresholds = np.zeros(length)
        self.__last_speech_detection_index = -1
        self.__silence_threshold_frame_size = 100 * self.__audio_frame_size
        self.__speech_detections = np.zeros(length)
        self.__tracking_index = int(length - self.__audio_frame_size / 2 - 1)
        self.__zero_crossings = np.zeros(length)
        self.__zero_crossings_threshold = 0

        # Callbacks
        self.__callbacks = []
        self.__callback_sleep_time = 0.025 # seconds
        self.__speech_segments = Queue()
        self.__stop_processing = False

        # Quality
        self.__play_samples = False
        self.__save_samples = False
        self.__save_samples_path = None

    def __audio_callback(self, indata, frames, time, status):
        data = indata[:, 0]
        shift = len(data)

        self.__data = np.roll(self.__data, -shift, axis = 0)
        self.__data[-shift:] = data[:]
        self.__tracking_index = self.__tracking_index - shift
        self.__last_speech_detection_index = self.__last_speech_detection_index - shift

        self.__energies = np.roll(self.__energies, -shift, axis = 0)
        self.__energies[-shift:] = 0
        self.__energies_max_thresholds = np.roll(self.__energies_max_thresholds, -shift, axis = 0)
        self.__energies_max_thresholds[-shift:] = 0
        self.__energies_min_thresholds = np.roll(self.__energies_min_thresholds, -shift, axis = 0)
        self.__energies_min_thresholds[-shift:] = 0
        self.__zero_crossings = np.roll(self.__zero_crossings, -shift, axis = 0)
        self.__zero_crossings[-shift:] = 0
        self.__speech_detections = np.roll(self.__speech_detections, -shift, axis = 0)
        self.__speech_detections[-shift:] = 0

        signal_samples_radius = self.__audio_frame_size / 2
        signal_start_index = self.__tracking_index
        signal_end_index = self.__tracking_index + shift

        signal_samples = self.__build_sample_matrix(self.__data, signal_start_index, signal_end_index, signal_samples_radius * 2)

        # Compute energy
        self.__energies[-shift:] = self.__calculate_energy(signal_samples)
        
        # Compute zero crossings    
        self.__zero_crossings[-shift:] = self.__calculate_zero_crossings(signal_samples)

        # Check for speech  
        if (self.__audio_frame_count >= self.__silence_threshold_frame_size) and (np.mod(self.__audio_frame_count, 5 * shift) == 0):        
            index = max(-int(3 * self.__fs), self.__last_speech_detection_index)
            energies = self.__energies[index:]
            zero_crossings = self.__zero_crossings[index:]
            energy_thresholds = self.__calculate_energy_threshold(energies)
            n1, n2 = self.__find_speech_segment(energies, zero_crossings, energy_thresholds[0], energy_thresholds[1], self.__zero_crossings_threshold, len(energies))
            if not np.isnan(n1):
                signal_data = self.__data[n1 : n2]
                if (signal_data.size != 0):
                    self.__speech_detections[n1 : n2] = np.max(signal_data)
                    self.__last_speech_detection_index = n2
                    self.__speech_segments.put(np.copy(signal_data))

        # Compute zero-crossing thresholds
        if (self.__audio_frame_count < self.__silence_threshold_frame_size) and ((self.__audio_frame_count + shift) >= self.__silence_threshold_frame_size):
            zero_crossings_samples = self.__zero_crossings[signal_end_index - self.__silence_threshold_frame_size : signal_end_index]
            self.__zero_crossings_threshold = self.__calculate_zero_crossing_threshold(zero_crossings_samples)

        self.__tracking_index = self.__tracking_index + shift
        self.__audio_frame_count = self.__audio_frame_count + shift

    def __build_sample_matrix(self, data, start_index, end_index, samples_per_column):
        matrix = np.zeros((samples_per_column, end_index - start_index))
        radius = samples_per_column / 2
        for i in range(start_index, end_index):
            matrix[:, i - start_index] = data[i - radius : i + radius]
        return matrix

    def __calculate_energy(self, data):
        return np.sum(np.abs(data), axis = 0)

    def __calculate_energy_threshold(self, energies):
        min_energy = np.min(energies)
        max_energy = np.max(energies)

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
            if isinstance(axes, np.ndarray) == True:
                ax = axes[plot_index]
            else:
                ax = axes
            self.__initialize_signal_plot(ax, self.__data[:], self.__speech_detections[:])
            plot_index = plot_index + 1
        if (not self.__hide_energy_plot):
            if isinstance(axes, np.ndarray) == True:
                ax = axes[plot_index]
            else:
                ax = axes
            self.__initialize_energy_plot(ax, self.__energies[:], self.__energies_min_thresholds[:], self.__energies_max_thresholds[:])
            plot_index = plot_index + 1
        if (not self.__hide_zero_crossing_plot):
            if isinstance(axes, np.ndarray) == True:
                ax = axes[plot_index]
            else:
                ax = axes
            self.__initialize_zero_crossings_plot(ax, self.__zero_crossings[:])
            plot_index = plot_index + 1
        if (not self.__hide_spectrogram_plot):
            if isinstance(axes, np.ndarray) == True:
                ax = axes[plot_index]
            else:
                ax = axes
            self.__initialize_spectrogram_plot(ax, self.__data[:])
            ax.axis('off')

        self.__fig.tight_layout(pad = 0)

    def __find_speech_segment(self, energies, zero_crossings, energy_min_threshold, energy_max_threshold, zero_crossing_threshold, max_distance):
        n1 = np.nan
        n2 = np.nan
        index = -1
        fricative_lookahead = int(0.25 * self.__fs)

        # Energy is really low, probably not speech, or they're whispering in which case they better learn to speak up...
        if np.max(energies) < 0.5:
            return np.nan, np.nan

        while index > -max_distance:

            # Find where we dip below the ITL from the end
            for i in range(-index - 1, max_distance):
                index = -i - 1
                self.__energies_min_thresholds[index] = energy_min_threshold
                self.__energies_max_thresholds[index] = energy_max_threshold
                if energies[index] < energy_min_threshold:
                    break

            if index <= -max_distance:
                return np.nan, np.nan

            # Find where we go above ITL from the end      
            for i in range(-index - 1, max_distance):
                index = -i - 1
                self.__energies_min_thresholds[index] = energy_min_threshold
                self.__energies_max_thresholds[index] = energy_max_threshold
                if energies[index] > energy_min_threshold:
                    break

            if index <= -max_distance:
                return np.nan, np.nan

            # If N2 was updated before, it must be greater than whatever index currently is
            if np.isnan(n2):
                n2 = index

            # Find where we exceed ITU
            for i in range(-index - 1, max_distance):
                index = -i - 1
                self.__energies_min_thresholds[index] = energy_min_threshold
                self.__energies_max_thresholds[index] = energy_max_threshold

                # Dipped under the min threshold before exceeding max
                if energies[index] < energy_min_threshold:
                    break
                if energies[index] > energy_max_threshold:
                    break

            if energies[index] < energy_min_threshold:
                continue
            if energies[index] > energy_max_threshold:
                break

        if index <= -max_distance:
            return np.nan, np.nan

        # Find where we dip below ITL
        for i in range(-index - 1, max_distance):
            index = -i - 1
            self.__energies_min_thresholds[index] = energy_min_threshold
            self.__energies_max_thresholds[index] = energy_max_threshold

            # Dipped under the min threshold
            if energies[index] < energy_min_threshold:
                break

        # Not enough data to search zero crossings in back
        if index - fricative_lookahead <= -max_distance:
            return np.nan, np.nan
        
        n1 = index
        index_start = n1
        index_end = n2
        zc_start_count = 0
        zc_end_count = 0

        # Look for fricatives
        for i in range(0, fricative_lookahead):
            zc_start = np.max(np.sign(zero_crossings[index_start - i] - zero_crossing_threshold), 0)
            zc_start_count = zc_start_count + zc_start
            zc_end = np.max(np.sign(zero_crossings[index_end + i] - zero_crossing_threshold), 0)
            zc_end_count = zc_end_count + zc_end

            # Move N1 to account for trailing fricatives
            if (zc_start == 1) and (zc_start_count >= 3):
                n1 = index_start - i
                self.__energies_min_thresholds[n1] = energy_min_threshold
                self.__energies_max_thresholds[n1] = energy_max_threshold
            # Move N2 to account for leading fricatives
            if (zc_end == 1) and (zc_end_count >= 3):
                n2 = index_end + i
                self.__energies_min_thresholds[n2] = energy_min_threshold
                self.__energies_max_thresholds[n2] = energy_max_threshold

        # Phrase is to short, most likely not speech
        if ((n2 - n1) / float(self.__fs)) < 0.25:
            return np.nan, np.nan
    
        # Adding 0.05s padding to start and end for silence
        n1 = n1 - 0.15 * self.__fs
        n2 = np.min(n2 + 0.15 * self.__fs, -1)

        return int(n1), int(n2)
    
    def __get_new_filepath(self, folder_path):
        i = 0
        template = os.path.join(folder_path, "%d.wav")
        while os.path.exists(template % i):
            i = i + 1
        return template % i

    def __initialize_energy_plot(self, ax, data, min_threshold_data, max_threshold_data):
        self.__energy_plot = ax
        self.__energy_plot_data, = ax.plot(self.__time, data)
        self.__energy_plot_max_data, = ax.plot(self.__time, max_threshold_data, color='g')
        self.__energy_plot_min_data, = ax.plot(self.__time, min_threshold_data, color='r')

        ax.axis((0, len(data), 0, 1))
        ax.set_title("Energy")
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        ax.set_xlabel("Time (s)")
        ax.set_xlim([-self.__window_duration, 0])
        ax.tick_params(bottom='on', top='off', labelbottom='on', right='off', left='on', labelleft='on')

    def __initialize_signal_plot(self, ax, signal_data, speech_detect_data):        
        self.__signal_plot = ax
        self.__signal_plot_data, = ax.plot(self.__time, signal_data)
        self.__signal_plot_speech_detect_data, = ax.plot(self.__time, speech_detect_data, color='r')
        self.__garbage_plot_data = ax.plot(self.__time, np.full(len(signal_data), np.nan))

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
        _,_,_,self.__spectrogram_plot_data = ax.specgram(data, NFFT = 1024, Fs = self.__fs, noverlap = 900)

    def __initialize_zero_crossings_plot(self, ax, data):
        self.__zero_crossings_plot = ax
        self.__zero_crossings_plot_data, = ax.plot(self.__time, data)
        self.__zero_crossings_plot_threshold_data, = ax.plot(self.__time, np.full(len(data), self.__zero_crossings_threshold), color='r')

        ax.axis((0, len(data), 0, 0.1))
        ax.set_title("Zero Crossings")
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        ax.set_xlabel("Time (s)")
        ax.set_xlim([-self.__window_duration, 0])
        ax.tick_params(bottom='on', top='off', labelbottom='on', right='off', left='on', labelleft='on')

    def __process_speech_segments(self):
        while not self.__stop_processing:
            time.sleep(self.__callback_sleep_time)

            if self.__speech_segments.empty():
                continue

            while not self.__speech_segments.empty():
                speech_segment = self.__speech_segments.get()

                for callback in self.__callbacks:
                    callback(speech_segment)

                if self.__save_samples:
                    with sf.SoundFile(self.__get_new_filepath(self.__save_samples_path), mode='x', samplerate = self.__fs, channels = 1) as file:
                        file.write(speech_segment)

                if self.__play_samples:
                    sd.play(speech_segment, self.__fs, blocking = True)
                
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

        self.__energy_plot_data.set_ydata(data)
        self.__energy_plot_max_data.set_ydata(max_threshold_data)
        self.__energy_plot_min_data.set_ydata(min_threshold_data)

    def __update_plots(self, frame):
        if self.__force_draw == True:
            self.__animation.event_source.stop()

        plot_data = []

        if ((not self.__hide_signal_plot) and (((time.time() - self.__last_updated_signal_time) > self.__signal_plot_refresh_interval) or self.__force_draw)):
            self.__update_signal_plot(self.__data[:], self.__speech_detections[:])
            self.__last_updated_signal_time = time.time()
            plot_data.append(self.__signal_plot_data)
            plot_data.append(self.__signal_plot_speech_detect_data)

        if ((not self.__hide_energy_plot) and (((time.time() - self.__last_updated_energy_time) > self.__energy_plot_refresh_interval) or self.__force_draw)):
            self.__update_energy_plot(self.__energies[:], self.__energies_min_thresholds[:], self.__energies_max_thresholds[:])
            self.__last_updated_energy_time = time.time()
            plot_data.append(self.__energy_plot_data)
            plot_data.append(self.__energy_plot_max_data)
            plot_data.append(self.__energy_plot_min_data)

        if ((not self.__hide_zero_crossing_plot) and (((time.time() - self.__last_updated_zero_crossings_time) > self.__zero_crossings_plot_refresh_interval) or self.__force_draw)):
            self.__update_zero_crossings_plot(self.__zero_crossings[:])
            self.__last_updated_zero_crossings_time = time.time()
            plot_data.append(self.__zero_crossings_plot_data)
            plot_data.append(self.__zero_crossings_plot_threshold_data)

        if ((not self.__hide_spectrogram_plot) and (((time.time() - self.__last_updated_spectrogram_time) > self.__spectrogram_plot_refresh_interval) or self.__force_draw)):
            self.__update_spectrogram_plot(self.__data[:])
            self.__last_updated_spectrogram_time = time.time()
            plot_data.append(self.__spectrogram_plot_data)

        if (self.__force_draw == True):
            self.__force_draw = False

        if (self.__last_pause_state == False) and (self.__pause == True):
            self.__force_draw = True

        self.__last_pause_state = self.__pause

        return plot_data

    def __update_signal_plot(self, signal_data, detect_data):
        self.__scale_plot(self.__signal_plot, signal_data, False)
        self.__signal_plot_data.set_ydata(signal_data)
        self.__signal_plot_speech_detect_data.set_ydata(detect_data)

    def __update_spectrogram_plot(self, data):
        self.__spectrogram_plot.clear()
        _,_,_,self.__spectrogram_plot_data = self.__spectrogram_plot.specgram(data, NFFT = 1024, Fs = self.__fs, noverlap = 900)
        self.__spectrogram_plot.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='on', labelleft='on')
        self.__spectrogram_plot.set_title("Spectrogram")

    def __update_zero_crossings_plot(self, data):
        threshold_data = np.full(len(data), self.__zero_crossings_threshold)

        self.__scale_plot(self.__zero_crossings_plot, data, False)
        self.__scale_plot(self.__zero_crossings_plot, threshold_data, True)

        self.__zero_crossings_plot_data.set_ydata(data)
        self.__zero_crossings_plot_threshold_data.set_ydata(threshold_data)

    def add_sample_callback(self, callback):
        self.__callbacks.append(callback)

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

    def play_samples(self, play = True):
        self.__play_samples = play

    def resume(self):
        if (self.__pause == True):
            self.__animation.event_source.start()
            self.__pause = False

    def run(self, show_plot = True): 
        sd.default.device["input"] = "Bose QuietComfort 35 H, MME"
        stream = sd.InputStream(channels=1, samplerate=self.__fs, callback=self.__audio_callback)
        callback_thread = Thread(target = self.__process_speech_segments)
        
        with stream:
            callback_thread.start()
            while True:
                if show_plot:
                    self.__create_plots()

                    # Need reference to animation otherwise garbage collector removes it...
                    self.__animation = animation.FuncAnimation(self.__fig, self.__update_plots, interval = self.__data_update_interval * 1000, blit = True, repeat = False)
                    plt.show()
                    self.__pause = False
                else:
                    time.sleep(1)
                
        self.__stop_processing = True
        callback_thread.join()

    def save_samples(self, path, save = True):
        self.__save_samples = save

        if save:
            if os.path.exists(path):
                self.__save_samples_path = path
            else:
                self.__save_samples = False
