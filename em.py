from asr_feature_builder import ASR_Feature_Builder
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io.wavfile as wavfile

class EM:

    def __init__(self):

        # Feature builder
        self.__feature_window_duration = 0.025 # seconds
        self.__feature_skip_duration = 0.01 # seconds
        self.__feature_nfilters = 26
        self.__feature_nfilters_keep = 13
        self.__feature_radius = 2
        self.__feature_builder = ASR_Feature_Builder()

    def __compute_ab_matrix(self, feature_matrix, hmm_parameters):
        mean_matrix = hmm_parameters.get_mean_matrix()
        nstates = hmm_parameters.get_nstates()
        transition_matrix = hmm_parameters.get_transition_matrix()
        variances_matrix = hmm_parameters.get_variances_matrix()

        a = np.zeros((nstates, feature_matrix.shape[1]))
        b = np.zeros(a.shape)

        a[0, 0] = self.__compute_gaussian_probability(feature_matrix[:, 0], mean_matrix[:, 0], variances_matrix[:, 0])
        b[-1, -1] = 1

        for t in range(1, a.shape[1]):
            for i in range(0, nstates - 1):
                t_a = t
                t_b = -t - 1
                i_a = i
                i_b = -i - 1
                j_a = i_a + 1
                j_b = i_b - 1

                a[j_a, t_a] = a[i_a, t_a - 1] * transition_matrix[i_a, j_a] + a[j_a, t_a - 1] * transition_matrix[j_a, j_a]
                if a[j_a, t_a] != 0:
                    p_a = self.__compute_gaussian_probability(feature_matrix[:, t_a], mean_matrix[:, j_a], variances_matrix[:, j_a])
                    a[j_a, t_a] = a[j_a, t_a] * p_a

                b[j_b, t_b] = b[i_b, t_b + 1] * transition_matrix[j_b, i_b] + b[j_b, t_b + 1] * transition_matrix[j_b, j_b]
                if b[j_b, t_b] != 0:
                    p_b = self.__compute_gaussian_probability(feature_matrix[:, t_b + 1], mean_matrix[:, i_b], variances_matrix[:, i_b])
                    b[j_b, t_b] = b[j_b, t_b] * p_b

            p_a = self.__compute_gaussian_probability(feature_matrix[:, t], mean_matrix[:, 0], variances_matrix[:, 0])
            a[0, t] = p_a * a[0, t - 1] * transition_matrix[0, 0]
            p_b = self.__compute_gaussian_probability(feature_matrix[:, -t], mean_matrix[:, 0], variances_matrix[:, 0])
            b[-1, -t - 1] = p_b * b[-1, -t] * transition_matrix[-1, -1]

            a[:, t] = np.true_divide(a[:, t], np.max(a[:, t]))
            b[:, -t - 1] = np.true_divide(b[:, -t - 1], np.max(b[:, -t - 1]))

        return a, b

    def __compute_gaussian_probability(self, feature_vector, mean_vector, variances_vector):
        exponent = np.multiply(-0.5, np.sum(np.true_divide(np.square(feature_vector - mean_vector), variances_vector)))
        denominator = np.power(2 * np.pi, len(feature_vector) / 2) * np.sqrt(np.prod(variances_vector))
        return np.exp(exponent) / float(denominator)

    def __initialize_hmm_parameters(self, nstates, feature_matrices):
        nfeatures = feature_matrices[0].shape[0]
        variances_matrix = np.zeros((nfeatures, nstates), dtype = np.float)
        mean_matrix = np.zeros((nfeatures, nstates), dtype = np.float)
        transition_matrix = np.zeros((nstates, nstates), dtype = np.float)

        for feature_matrix in feature_matrices:
            variances = np.square(np.std(feature_matrix, axis = 1))
            means = np.mean(feature_matrix, axis = 1)
            variances_matrix[:, 0] = np.add(variances_matrix[:, 0], variances)
            mean_matrix[:, 0] = np.add(mean_matrix[:, 0], means)
        
        variances_matrix[:, 0] = np.true_divide(variances_matrix[:, 0], len(feature_matrices))
        mean_matrix[:, 0] = np.true_divide(mean_matrix[:, 0], len(feature_matrices))

        for i in range(1, nstates):
            variances_matrix[:, i] = variances_matrix[:, 0]
            mean_matrix[:, i] = mean_matrix[:, 0]

        for i in range(0, nstates - 1):
            transition_matrix[i, i] = 0.5
            transition_matrix[i, i + 1] = 0.5
        transition_matrix[-1, -1] = 1

        return HMM_Parameters(nstates, transition_matrix, mean_matrix, variances_matrix)

    def build_hmm_from_folder(self, folder_path, nstates):
        audio_files = []

        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                audio_files.append(file_path)
        
        return self.build_hmm_from_files(audio_files, nstates)

    def build_hmm_from_files(self, audio_files, nstates):
        feature_matrices = []

        for audio_file in audio_files:
            fs, s = wavfile.read(audio_file, 'rb')
            feature_matrix = self.__feature_builder.compute_features_for_signal( \
                s, \
                fs, \
                self.__feature_nfilters, \
                self.__feature_window_duration, \
                self.__feature_skip_duration, \
                self.__feature_radius, \
                self.__feature_nfilters_keep)

            feature_matrices.append(feature_matrix)
        
        return self.build_hmm_from_feature_matrices(feature_matrices, nstates)

    def build_hmm_from_feature_matrices(self, feature_matrices, nstates):
        hmm_parameters = self.__initialize_hmm_parameters(nstates, feature_matrices)

        for feature_matrix in feature_matrices:
            a, b = self.__compute_ab_matrix(feature_matrix, hmm_parameters)
            plt.figure()
            plt.imshow(a, aspect='auto')
            plt.show()
        
        return None

class HMM_Parameters:

    def __init__(self, nstates, transition_matrix, mean_matrix, variances_matrix):
        self.__variances_matrix = variances_matrix
        self.__mean_matrix = mean_matrix
        self.__nstates = nstates
        self.__transition_matrix = transition_matrix
    
    def get_variances_matrix(self):
        return self.__variances_matrix
    
    def get_mean_matrix(self):
        return self.__mean_matrix

    def get_nstates(self):
        return self.__nstates
    
    def get_transition_matrix(self):
        return self.__transition_matrix

if __name__ == '__main__':
    folder_path = "C:\Users\AkramAsylum\OneDrive\Courses\School\EE 516 - Compute Speech Processing\Assignments\Assignment 5\samples\odessa"

    em = EM()
    em.build_hmm_from_folder(folder_path, 10)