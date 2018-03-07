import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct

class ASR_Feature_Builder:

    def __init__(self):
        self.__all_plot_filter_bank_indices = None
        self.__blocking = True
        self.__delta_features_matrix = None
        self.__features_matrix = None
        self.__fft_freq = None
        self.__fft_window_length = 0
        self.__fft_mag = None
        self.__filter_banks = None
        self.__filtered_spectra = None
        self.__filtered_spectra_sums = None
        self.__filtered_spectra_sums_log = None
        self.__fs = 0
        self.__mfcc_features_matrix = None
        self.__nfilters = 0
        self.__nwindows = 0
        self.__signal = None
        self.__skip_size = 0
        self.__window_size = 0

    def __compute_mfcc_for_window(self, s, window_index):
        self.__fft_mag[window_index, :] = np.abs(np.fft.fft(s, n = self.__fft_window_length))      

        for i in range(0, self.__nfilters):
            self.__filtered_spectra[window_index, i, :] = np.multiply(self.__filter_banks[i, :], self.__fft_mag[window_index, :])
            self.__filtered_spectra_sums[window_index, i] = np.sum(self.__filtered_spectra[window_index, i, :])

        self.__filtered_spectra_sums_log[window_index, :] = np.log(self.__filtered_spectra_sums[window_index, :])

        return dct(self.__filtered_spectra_sums_log[window_index, :])

    def __create_filter_banks(self, freq, lower_freq, upper_freq):
        npoints = self.__nfilters + 2
        mel_freq = np.linspace(self.__frequency2mel(lower_freq), self.__frequency2mel(upper_freq), npoints)
        filter_bank_freq = self.__mel2frequency(mel_freq)
        filter_bank_freq_indices = np.floor(np.divide(np.multiply(self.__fft_window_length + 1, filter_bank_freq), self.__fs)).astype(int)
        filter_banks = np.zeros(shape = (self.__nfilters, self.__fft_window_length), dtype=float)
        
        for i in range(0, filter_bank_freq_indices.size - 2):
            start_index = filter_bank_freq_indices[i]
            mid_index = filter_bank_freq_indices[i + 1]
            end_index = filter_bank_freq_indices[i + 2]

            for j in range(start_index, mid_index):
                filter_banks[i, j] = (freq[j] - freq[start_index]) / (freq[mid_index] - freq[start_index])

            for j in range(mid_index, end_index):
                filter_banks[i, j] = 1 - (freq[j] - freq[mid_index]) / (freq[end_index] - freq[mid_index])
        
        return filter_banks

    def __frequency2mel(self, f):
        return np.multiply(1125, np.log(1 + np.true_divide(f, 700)))

    def __mel2frequency(self, m):
        return np.multiply(700, np.exp(np.true_divide(m, 1125)) - 1)

    def compute_delta(self, filepath, nfilters, window_duration, skip_duration, radius):
        fs, s = wavfile.read(filepath, 'rb')

        return self.compute_delta_for_signal(s, fs, nfilters, window_duration, skip_duration, radius)

    def compute_delta_for_signal(self, s, fs, nfilters, window_duration, skip_duration, radius):
        mfcc_matrix = self.compute_mfcc_for_signal(s, fs, nfilters, window_duration, skip_duration)
        self.__delta_features_matrix = np.zeros(shape = (self.__nfilters, self.__nwindows), dtype=float)

        divisor = 0
        for i in range(-radius, radius + 1):
            divisor = divisor + (i ** 2)

        for i in range(0, self.__nfilters):
            for j in range(radius, self.__nwindows - radius):
                for k in range(-radius, radius + 1):
                    self.__delta_features_matrix[i, j] = self.__delta_features_matrix[i, j] + k * mfcc_matrix[i, j + k]
            self.__delta_features_matrix[i, j] = self.__delta_features_matrix[i, j] / float(divisor)
        
        self.__delta_features_matrix = self.__delta_features_matrix[:, radius : -radius]

        return self.__delta_features_matrix

    def compute_features(self, filepath, nfilters, window_duration, skip_duration, radius, nfilters_to_keep):
        fs, s = wavfile.read(filepath, 'rb')

        return self.compute_features_for_signal(s, fs, nfilters, window_duration, skip_duration, radius, nfilters_to_keep)

    def compute_features_for_signal(self, s, fs, nfilters, window_duration, skip_duration, radius, nfilters_to_keep):     
        self.compute_delta_for_signal(s, fs, nfilters, window_duration, skip_duration, radius)

        mfcc_matrix = self.__mfcc_features_matrix[range(0, nfilters_to_keep), :]
        delta_features_matrix = self.__delta_features_matrix[range(0, nfilters_to_keep), :]
        self.__features_matrix = np.concatenate((mfcc_matrix[:, radius : -radius], delta_features_matrix), axis = 0)

        return self.__features_matrix

    def compute_mfcc(self, filepath, nfilters, window_duration, skip_duration):
        fs, s = wavfile.read(filepath, 'rb')

        return self.compute_mfcc_for_signal(s, fs, nfilters, window_duration, skip_duration)

    def compute_mfcc_for_signal(self, s, fs, nfilters, window_duration, skip_duration):
        offset = 0
        frame_index = 0
        next_power = 1

        self.__fs = fs
        self.__nfilters = nfilters
        self.__signal = s
        self.__skip_size = int(skip_duration * self.__fs)
        self.__window_size = int(window_duration * self.__fs)
        self.__all_plot_filter_bank_indices = range(0, self.__nfilters)
        self.__nwindows = int(np.ceil((s.size - self.__window_size) / float(self.__skip_size)))
        self.__fft_window_length = int(np.power(2, (next_power-1) + np.ceil(np.log2(self.__window_size))))
        self.__fft_freq = np.fft.fftfreq(self.__fft_window_length) * self.__fs
        self.__mfcc_features_matrix = np.ndarray(shape=(self.__nfilters, self.__nwindows), dtype=float)
        self.__fft_mag = np.ndarray(shape = (self.__nwindows, self.__fft_window_length), dtype=float)
        self.__filtered_spectra = np.ndarray(shape = (self.__nwindows, self.__nfilters, self.__fft_window_length), dtype=float)
        self.__filtered_spectra_sums = np.ndarray(shape = (self.__nwindows, self.__nfilters), dtype=float)
        self.__filtered_spectra_sums_log = np.ndarray(shape = (self.__nwindows, self.__nfilters), dtype=float)

        self.__filter_banks = self.__create_filter_banks(self.__fft_freq, 0, np.max(self.__fft_freq))

        while (offset + self.__window_size) < (s.size - self.__window_size):
            frame_index = offset / self.__skip_size
            self.__mfcc_features_matrix[:, frame_index] = self.__compute_mfcc_for_window(s[offset : (offset + self.__window_size)], frame_index)
            offset = offset + self.__skip_size
    
        # Side effects of performing DCT can result in dividing by zero errors, replace these values with the median.ASR_Feature_Builder
        # Fixing this clears up resultant plots as well since the color scheme isn't skewed by large magnitudes
        # Should probably do proper outlier detection here instead of checking against an arbitrary threshold
        self.__mfcc_features_matrix[np.where(np.abs(self.__mfcc_features_matrix) > 10000)] = np.median(self.__mfcc_features_matrix)

        return self.__mfcc_features_matrix

    def filter_all_plot_filter_bank_indices(self, bank_indices):
        self.__all_plot_filter_bank_indices = bank_indices

    def get_filter_bank_count(self):
        return self.__nfilters

    def get_window_count(self):
        return self.__nwindows

    def plot_all_delta_features_matrix(self):
        self.plot_delta_features_matrix(len(self.__all_plot_filter_bank_indices))

    def plot_all_filter_banks(self):
        self.plot_filter_bank(self.__all_plot_filter_bank_indices)

    def plot_all_filtered_spectra(self, window_index):
        self.plot_filtered_spectra(self.__all_plot_filter_bank_indices, window_index)

    def plot_all_filtered_spectra_mfcc(self):
        self.plot_filtered_spectra_mfcc(self.__all_plot_filter_bank_indices)

    def plot_all_filtered_spectra_sum(self):
        self.plot_filtered_spectra_sum(self.__all_plot_filter_bank_indices)

    def plot_all_filtered_spectra_sum_log(self):
        self.plot_filtered_spectra_sum_log(self.__all_plot_filter_bank_indices)       

    def plot_all_mfcc_features_matrix(self):
        self.plot_mfcc_features_matrix(len(self.__all_plot_filter_bank_indices))

    def plot_delta_features_matrix(self, nfilters):
        plt.figure()
        plt.imshow(self.__delta_features_matrix[0:nfilters, :], aspect='auto')
        plt.title("Delta features matrix for filter banks 0:" + str(nfilters - 1))
        plt.xlabel('windows')
        plt.ylabel('delta')
        plt.show(block = self.__blocking)

    def plot_features_matrix(self):
        plt.figure()
        plt.imshow(self.__features_matrix, aspect='auto')
        plt.title("Features matrix")
        plt.xlabel('windows')
        plt.ylabel('features')
        plt.show(block = self.__blocking)

    def plot_filter_bank(self, bank_indices):
        plt.figure()
        
        for i in bank_indices:
            plt.plot(self.__fft_freq, self.__filter_banks[i, :], label='Filter Bank:%d' % i)

        plt.title("Filter banks")
        plt.xlabel('frequency (hz)')
        plt.ylabel('amplitude')
        plt.xlim([0, np.max(self.__fft_freq)])
        plt.legend()
        plt.show(block = self.__blocking)

    def plot_filtered_spectra(self, bank_indices, window_index):
        plt.figure()

        for i in bank_indices:
            plt.plot(self.__fft_freq, self.__filtered_spectra[window_index, i, :], label='Filter Bank:%d' % i)

        plt.title("Filtered spectra for window " + str(window_index) + " by filter bank")
        plt.xlabel('frequency (hz)')
        plt.ylabel('amplitude')
        plt.xlim([0, np.max(self.__fft_freq)])
        plt.legend()
        plt.show(block = self.__blocking)

    def plot_filtered_spectra_mfcc(self, bank_indices):
        plt.figure()

        for i in bank_indices:
            plt.plot(range(0, self.__nwindows), self.__mfcc_features_matrix[i, :], label='Filter Bank:%d' % i)

        plt.title("Filtered spectra mfcc")
        plt.xlabel('windows')
        plt.ylabel('dct(log(sum))')
        plt.legend()
        plt.show(block = self.__blocking)

    def plot_filtered_spectra_sum(self, bank_indices):
        plt.figure()

        for i in bank_indices:
            plt.plot(range(0, self.__nwindows), self.__filtered_spectra_sums[:, i], label='Filter Bank:%d' % i)

        plt.title("Filtered spectra sum")
        plt.xlabel('windows')
        plt.ylabel('sum')
        plt.legend()
        plt.show(block = self.__blocking)

    def plot_filtered_spectra_sum_log(self, bank_indices):
        plt.figure()

        for i in bank_indices:
            plt.plot(range(0, self.__nwindows), self.__filtered_spectra_sums_log[:, i], label='Filter Bank:%d' % i)

        plt.title("Filtered spectra log(sum)")
        plt.xlabel('windows')
        plt.ylabel('log(sum)')
        plt.legend()
        plt.show(block = self.__blocking)

    def plot_mfcc_features_matrix(self, nfilters):
        plt.figure()
        plt.imshow(self.__mfcc_features_matrix[0:nfilters, :], aspect='auto')
        plt.title("MFCC matrix for coefficients 0:" + str(nfilters - 1))
        plt.xlabel('windows')
        plt.ylabel('coefficients')
        plt.show(block = self.__blocking)

    def plot_mfcc_transitions(self, nfilters):
        plt.figure()
        for i in range(0, self.__nwindows):
            plt.plot(range(0, nfilters), self.__mfcc_features_matrix[0:nfilters, i])
        plt.title("MFCC transitions for all windows")
        plt.xlabel('index')
        plt.ylabel('coefficients')
        plt.show(block = self.__blocking)

    def plot_signal(self):
        plt.figure()
        duration = self.__signal.size / float(self.__fs)
        t = np.linspace(0, duration, self.__signal.size)

        plt.plot(t, self.__signal)
        plt.title("Audio signal")
        plt.xlabel('time (s)')
        plt.show(block = self.__blocking)     

    def set_plot_blocking(self, blocking):
        self.__blocking = blocking