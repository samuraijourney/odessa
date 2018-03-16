from asr_feature_builder import ASR_Feature_Builder
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.io.wavfile as wavfile

class HMM_Parameters:

    def __init__(self, nstates, transition_matrix, mean_matrix, variance_matrix, data_log_likelihood):
        self.__data_log_likelihood = data_log_likelihood
        self.__mean_matrix = mean_matrix
        self.__nstates = nstates
        self.__transition_matrix = transition_matrix
        self.__variance_matrix = variance_matrix

    def get_data_log_likelihood(self):
        return self.__data_log_likelihood

    def get_mean_matrix(self):
        return self.__mean_matrix

    def get_nstates(self):
        return self.__nstates
    
    def get_transition_matrix(self):
        return self.__transition_matrix

    def get_variance_matrix(self):
        return self.__variance_matrix

class HMM:

    def __init__(self):
        self.__feature_window_duration = 0.025 # seconds
        self.__feature_skip_duration = 0.01 # seconds
        self.__feature_nfilters = 26
        self.__feature_nfilters_keep = 13
        self.__feature_radius = 2
        self.__viterbi_path = None

    def __compute_gaussian_probability_log(self, feature_vector):
        feature_matrix = np.transpose(np.tile(feature_vector, (self.__nstates, 1)))
        exponent = -0.5 * np.sum(np.true_divide(np.square(feature_matrix - self.__mean_matrix), self.__variance_matrix), axis = 0)
        denominator = 0.5 * self.__nfeatures * np.log(2 * np.pi) + 0.5 * np.sum(np.log(self.__variance_matrix), axis = 0)
        return exponent - denominator

    def initialize_from_data(self, transition_matrix, mean_matrix, variance_matrix):
        self.__mean_matrix = mean_matrix
        self.__transition_matrix = transition_matrix
        self.__variance_matrix = variance_matrix

        self.__nstates = self.__transition_matrix.shape[0]
        self.__nfeatures = self.__mean_matrix.shape[0]

    def initialize_from_file(self, file_path):
        file_handle = open(file_path, 'rb')
        hmm = pickle.load(file_handle)
        file_handle.close()

        self.initialize_from_data(hmm.__transition_matrix, hmm.__mean_matrix, hmm.__variance_matrix)

    def initialize_from_hmm_parameters(self, hmm_parameters):
        self.initialize_from_data(hmm_parameters.get_transition_matrix(), hmm_parameters.get_mean_matrix(), hmm_parameters.get_variance_matrix())

    def match_from_feature_matrices(self, feature_matrices):
        if not isinstance(feature_matrices, list):
            feature_matrices = [feature_matrices]

        matches = []

        for feature_matrix in feature_matrices:
            self.__viterbi_path = np.zeros((self.__nstates, feature_matrix.shape[1]))
            match = self.__compute_gaussian_probability_log(feature_matrix[:, 0])[0]
            current_state = 0
            self.__viterbi_path[current_state, 0] = 1

            for t in range(1, feature_matrix.shape[1]):
                p = self.__compute_gaussian_probability_log(feature_matrix[:, t])
                log_probabilities = self.__transition_matrix[current_state, :] + p
                current_state = np.argmax(log_probabilities)
                self.__viterbi_path[current_state, t] = 1
                match = match + log_probabilities[current_state]
            
            matches.append(match)

        return matches

    def match_from_feature_matrix(self, feature_matrix):
        return self.match_from_feature_matrices(feature_matrix)[0]

    def match_from_files(self, audio_files):
        if not isinstance(audio_files, list):
            audio_files = [audio_files]

        signals = []
        fs = -1

        for audio_file in audio_files:
            fs, s = wavfile.read(audio_file, 'rb')
            signals.append(s)
        
        return self.match_from_signals(signals, fs)

    def match_from_folder(self, folder_path):
        audio_files = []

        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                audio_files.append(file_path)
        
        return self.match_from_files(audio_files)

    def match_from_signals(self, signals, fs):
        if not isinstance(signals, list):
            signals = [signals]

        feature_matrices = []
        feature_builder = ASR_Feature_Builder()

        for s in signals:
            feature_matrix = feature_builder.compute_features_for_signal( \
                s, \
                fs, \
                self.__feature_nfilters, \
                self.__feature_window_duration, \
                self.__feature_skip_duration, \
                self.__feature_radius, \
                self.__feature_nfilters_keep)

            feature_matrices.append(feature_matrix)
        
        return self.match_from_feature_matrices(feature_matrices)

    def plot_viterbi_path(self):
        plt.figure()
        plt.imshow(self.__viterbi_path, aspect='auto')
        plt.title("Viterbi path")
        plt.xlabel('time')
        plt.ylabel('states')
        plt.show()

    def save(self, file_path):
        file_handle = open(file_path, 'wb')
        pickle.dump(self, file_handle)
        file_handle.close()