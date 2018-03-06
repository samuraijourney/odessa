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

        # Continuous builder
        self.__speech_segments = []

    def __compute_ab_matrix(self, feature_matrix, hmm_parameters):
        mean_matrix = hmm_parameters.get_mean_matrix()
        nstates = hmm_parameters.get_nstates()
        transition_matrix = hmm_parameters.get_transition_matrix()
        variance_matrix = hmm_parameters.get_variance_matrix()

        a = np.zeros((nstates, feature_matrix.shape[1]))
        b = np.zeros(a.shape)

        a[0, 0] = 1
        b[:, -1] = 1 / float(len(b[:, -1]))

        for t in range(1, a.shape[1]):

            a[:, t] = np.multiply( \
                np.dot( \
                    np.transpose(transition_matrix), \
                    a[:, t - 1] \
                ), \
                self.__compute_gaussian_probability(feature_matrix[:, t], mean_matrix, variance_matrix) \
            )

            b[:, -t - 1] = np.multiply(
                np.dot( \
                    np.transpose(transition_matrix), \
                    b[:, -t] \
                ), \
                self.__compute_gaussian_probability(feature_matrix[:, -t], mean_matrix, variance_matrix) \
            )

            a[:, t] = np.true_divide(a[:, t], np.sum(a[:, t]))
            b[:, -t - 1] = np.true_divide(b[:, -t - 1], np.sum(b[:, -t - 1]))

        return a, b

    def __compute_gaussian_probability(self, feature_vector, mean_matrix, variances_matrix):
        nstates = mean_matrix.shape[1]
        feature_matrix = self.__convert_vector_to_matrix(feature_vector, nstates)
        exponent = np.multiply(-0.5, np.sum(np.true_divide(np.square(feature_matrix - mean_matrix), variances_matrix), axis = 0))
        denominator = np.multiply(np.power(2 * np.pi, nstates / 2), np.sqrt(np.prod(variances_matrix, axis = 0)))
        return np.true_divide(np.exp(exponent), denominator)

    def __compute_gz(self, a, b, feature_matrix, hmm_parameters):
        mean_matrix = hmm_parameters.get_mean_matrix()
        transition_matrix = hmm_parameters.get_transition_matrix()
        variance_matrix = hmm_parameters.get_variance_matrix()

        g = np.zeros(a.shape)
        z = np.zeros(a.shape)

        ab = np.multiply(a[:, 0], b[:, 0])
        g[:, 0] = np.true_divide(ab, np.sum(ab))

        for t in range(1, z.shape[1]):
            ab = np.multiply(a[:, t], b[:, t])
            g[:, t] = np.true_divide(ab, np.sum(ab))

            z[:, t] = np.multiply( \
                np.multiply( \
                    np.dot( \
                        transition_matrix,
                        a[:, t - 1] \
                    ), \
                    b[:, t] \
                ), \
                self.__compute_gaussian_probability(feature_matrix[:, t], mean_matrix, variance_matrix) \
            )

        g = np.true_divide(g, np.max(g, axis = 0))
        z = np.true_divide(z, np.max(z, axis = 0))
        
        return g, z

    def __convert_vector_to_matrix(self, vector, ncolumns):
        return np.transpose(np.tile(vector, (ncolumns, 1)))

    def __initialize_hmm_parameters(self, nstates, feature_matrices):
        nfeatures = feature_matrices[0].shape[0]
        variance_vector = np.zeros(nfeatures, dtype = np.float)
        mean_vector = np.zeros(nfeatures, dtype = np.float)
        transition_matrix = np.zeros((nstates, nstates), dtype = np.float)

        for feature_matrix in feature_matrices:
            variance_vector = np.add(variance_vector, np.square(np.std(feature_matrix, axis = 1)))
            mean_vector = np.add(mean_vector, np.mean(feature_matrix, axis = 1))    
        variance_vector = np.true_divide(variance_vector, len(feature_matrices))
        mean_vector = np.true_divide(mean_vector, len(feature_matrices))

        variance_matrix = self.__convert_vector_to_matrix(variance_vector, nstates)
        mean_matrix = self.__convert_vector_to_matrix(mean_vector, nstates)

        mean_noise_matrix = np.random.rand(nfeatures, nstates)
        variance_noise_matrix = np.random.rand(nfeatures, nstates)

        for i in range(0, nfeatures):
            mean_scale = np.sqrt(variance_vector[i]) / 16.0
            variance_scale = variance_vector[i] / 16.0
            mean_noise_matrix[i, :] = mean_scale * mean_noise_matrix[i, :]
            variance_noise_matrix[i, :] = variance_scale * variance_noise_matrix[i, :]

        #mean_matrix = np.add(mean_matrix, mean_noise_matrix)
        #variance_matrix = np.add(variance_matrix, variance_noise_matrix)

        for i in range(0, nstates - 1):
            #stay_probability = np.random.uniform(0, 1, 1)[0]
            stay_probability = 0.5
            transition_probability = 1 - stay_probability
            transition_matrix[i, i] = transition_probability
            transition_matrix[i, i + 1] = 1 - transition_probability

        # At the end the probability of staying is 100% since it is the end of the HMM
        transition_matrix[-1, -1] = 1 

        return HMM_Parameters(nstates, transition_matrix, mean_matrix, variance_matrix)

    def build_hmm_from_folder(self, folder_path, nstates):
        audio_files = []

        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                audio_files.append(file_path)
        
        return self.build_hmm_from_files(audio_files, nstates)

    def build_hmm_from_files(self, audio_files, nstates):
        signals = []
        fs = -1

        for audio_file in audio_files:
            fs, s = wavfile.read(audio_file, 'rb')
            signals.append(s)
        
        return self.build_hmm_from_signals(signals, fs, nstates)

    def build_hmm_from_feature_matrices(self, feature_matrices, nstates):
        hmm_parameters = self.__initialize_hmm_parameters(nstates, feature_matrices)

        for feature_matrix in feature_matrices:
            self.__a, self.__b = self.__compute_ab_matrix(feature_matrix, hmm_parameters)
            self.__g, self.__z = self.__compute_gz(self.__a, self.__b, feature_matrix, hmm_parameters)
            
            self.plot_all_matrices()
        
        return None

    def build_hmm_from_signals(self, signals, fs, nstates):
        feature_matrices = []

        for s in signals:
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

    def build_hmm_continuous(self, speech_segment, fs, nstates):
        self.__speech_segments.append(speech_segment)

        return self.build_hmm_from_signals(self.__speech_segments, fs, nstates)

    def plot_all_matrices(self, show = True):
        self.plot_alpha_beta_multiply_sum(False)
        self.plot_alpha_matrix(False)
        self.plot_beta_matrix(False)
        #self.plot_gamma_matrix(False)
        #self.plot_zeta_matrix(False)

        if show == True:
            plt.show()

    def plot_alpha_beta_multiply_sum(self, show = True):
        plt.figure()
        plt.imshow([np.sum(np.multiply(self.__a, self.__b), axis = 0)], aspect='auto')
        plt.title("Sum alpha * beta vector")
        plt.xlabel('frames')
        plt.ylabel('states')
        
        if show == True:
            plt.show()

    def plot_alpha_matrix(self, show = True):
        plt.figure()
        plt.imshow(self.__a, aspect='auto')
        plt.title("Alpha matrix")
        plt.xlabel('frames')
        plt.ylabel('states')
        
        if show == True:
            plt.show()

    def plot_beta_matrix(self, show = True):
        plt.figure()
        plt.imshow(self.__b, aspect='auto')
        plt.title("Beta matrix")
        plt.xlabel('frames')
        plt.ylabel('states')
        
        if show == True:
            plt.show()

    def plot_gamma_matrix(self, show = True):
        plt.figure()
        plt.imshow(self.__g, aspect='auto')
        plt.title("Gamma matrix")
        plt.xlabel('frames')
        plt.ylabel('states')
        
        if show == True:
            plt.show()

    def plot_zeta_matrix(self, show = True):
        plt.figure()
        plt.imshow(self.__z, aspect='auto')
        plt.title("Zeta matrix")
        plt.xlabel('frames')
        plt.ylabel('states')
        
        if show == True:
            plt.show()

    #def train_hmm_live(self):

class HMM_Parameters:

    def __init__(self, nstates, transition_matrix, mean_matrix, variance_matrix):
        self.__variance_matrix = variance_matrix
        self.__mean_matrix = mean_matrix
        self.__nstates = nstates
        self.__transition_matrix = transition_matrix
    
    def get_variance_matrix(self):
        return self.__variance_matrix
    
    def get_mean_matrix(self):
        return self.__mean_matrix

    def get_nstates(self):
        return self.__nstates
    
    def get_transition_matrix(self):
        return self.__transition_matrix

if __name__ == '__main__':
    folder_path = "C:\Users\AkramAsylum\OneDrive\Courses\School\EE 516 - Compute Speech Processing\Assignments\Assignment 5\samples\odessa"

    em = EM()
    em.build_hmm_from_folder(folder_path, 40)