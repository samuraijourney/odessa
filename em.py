from asr_feature_builder import ASR_Feature_Builder
from matplotlib import animation
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import os
import Queue
import scipy.io.wavfile as wavfile

class EM:

    def __init__(self):

        self.__log_zero = -10e30

        # Feature builder
        self.__feature_window_duration = 0.025 # seconds
        self.__feature_skip_duration = 0.01 # seconds
        self.__feature_nfilters = 26
        self.__feature_nfilters_keep = 13
        self.__feature_radius = 2
        self.__feature_builder = ASR_Feature_Builder()

        # Continuous builder
        self.__speech_segments = []

    def __compute_ab_matrices(self, feature_matrix, hmm_parameters):
        mean_matrix = hmm_parameters.get_mean_matrix()
        nstates = hmm_parameters.get_nstates()
        transition_matrix = hmm_parameters.get_transition_matrix()
        variance_matrix = hmm_parameters.get_variance_matrix()

        a = np.full((nstates, feature_matrix.shape[1]), self.__log_zero)
        b = np.full(a.shape, self.__log_zero)

        #a[:, 0] = hmm_parameters.get_initial_state_vector()
        a[0, 0] = 1
        b[-1, -1] = np.log(1)

        for t in range(1, a.shape[1]):
            p_a = self.__compute_gaussian_probability_log(feature_matrix[:, t], mean_matrix, variance_matrix)
            p_b = self.__compute_gaussian_probability_log(feature_matrix[:, -t], mean_matrix, variance_matrix)
            for j in range(0, nstates):
                a[j, t] = self.__sum_log_probabilities(a[:, t - 1] + transition_matrix[:, j]) + p_a[j]
                b[j, -t - 1] = self.__sum_log_probabilities(b[:, -t] + transition_matrix[j, :] + p_b)

        return a, b

    def __compute_data_log_likelihood(self, a_matrices):
        p = 1
        for i in range(0, len(a_matrices)):
            p = p * a_matrices[i][-1, -1]
        return p

    def __compute_gaussian_probability_log(self, feature_vector, mean_matrix, variance_matrix):
        nstates = mean_matrix.shape[1]
        nfeatures = len(feature_vector)
        feature_matrix = self.__convert_vector_to_matrix(feature_vector, nstates)
        exponent = -0.5 * np.sum(np.true_divide(np.square(feature_matrix - mean_matrix), variance_matrix), axis = 0)
        denominator = -0.5 * nfeatures * np.log(2 * np.pi) - 0.5 * np.sum(np.log(variance_matrix), axis = 0)
        return exponent + denominator

    def __compute_gz_matrices(self, a, b, feature_matrix, hmm_parameters):
        mean_matrix = hmm_parameters.get_mean_matrix()
        transition_matrix = hmm_parameters.get_transition_matrix()
        variance_matrix = hmm_parameters.get_variance_matrix()
        nstates = a.shape[0]
        nframes = a.shape[1]

        g = np.full(a.shape, self.__log_zero)
        z = np.full((nstates, nstates, nframes), self.__log_zero)

        g[:, 0] = a[:, 0] + b[:, 0] - self.__sum_log_probabilities(a[:, 0] + b[:, 0])

        for t in range(1, nframes):
            g[:, t] = a[:, t] + b[:, t] - self.__sum_log_probabilities(a[:, t] + b[:, t])

            p = self.__compute_gaussian_probability_log(feature_matrix[:, t], mean_matrix, variance_matrix)

            for q2 in range(0, nstates):
                for q1 in range(0, nstates):
                    z[q1, q2, t] = b[q2, t] + a[q1, t - 1] + transition_matrix[q1, q2] + p[q2]

        return g, z

    def __compute_new_hmm_parameters(self, feature_matrices, hmm_parameters):
        a_matrices = []
        b_matrices = []
        g_matrices = []
        z_matrices = []

        for feature_matrix in feature_matrices:
            a, b = self.__compute_ab_matrices(feature_matrix, hmm_parameters)
            g, z = self.__compute_gz_matrices(a, b, feature_matrix, hmm_parameters)
            a_matrices.append(a)
            b_matrices.append(b)
            g_matrices.append(g)
            z_matrices.append(z)
        
        self.__a = a
        self.__b = b
        self.__g = g
        self.__z = z

        new_initial_state_vector = self.__compute_new_initial_state_vector(a_matrices, b_matrices)
        new_transition_matrix = self.__compute_new_state_transition_matrix(g_matrices, z_matrices)
        new_mean_matrix = self.__compute_new_mean_matrix(feature_matrices, g_matrices)
        new_variance_matrix = self.__compute_new_variance_matrix(feature_matrices, new_mean_matrix, g_matrices)
        data_log_likelihood = self.__compute_data_log_likelihood(a_matrices)
        
        return HMM_Parameters(hmm_parameters.get_nstates(), new_initial_state_vector, new_transition_matrix, new_mean_matrix, new_variance_matrix, data_log_likelihood)

    def __compute_new_initial_state_vector(self, a_matrices, b_matrices):
        nstates = a_matrices[0].shape[0]
        nmatrices = len(a_matrices)
        results = np.full((nmatrices, nstates), self.__log_zero)
        initial_state_vector = np.full(nstates, self.__log_zero)

        for i in range(0, len(a_matrices)):
            a = a_matrices[i]
            b = b_matrices[i]
            results[i, :] = a[:, 0] + b[:, 0] - self.__sum_log_probabilities(a[:, 0] + b[:, 0])
        initial_state_vector = self.__sum_log_probability_matrix(results)
        
        return initial_state_vector - np.log(nmatrices)

    def __compute_new_mean_matrix(self, feature_matrices, g_matrices):
        nstates = g_matrices[0].shape[0]
        nfeatures = feature_matrices[0].shape[0]
        mean_matrix = np.zeros((nfeatures, nstates), dtype = np.float)
        numerator = np.zeros(nfeatures)
        denominator = 0.0

        for q in range(0, nstates):
            for i in range(0, len(feature_matrices)):
                feature_matrix = feature_matrices[i]
                g = g_matrices[i]
                numerator = numerator + np.sum(np.multiply(feature_matrix, np.exp(g[q, :])), axis = 1)
                denominator = denominator + np.sum(np.exp(g[q, :]))
            mean_matrix[:, q] = numerator / float(denominator)
        
        return mean_matrix

    def __compute_new_state_transition_matrix(self, g_matrices, z_matrices):
        nstates = g_matrices[0].shape[0]
        nmatrices = len(g_matrices)
        numerator = np.full((nstates, nstates, nmatrices), self.__log_zero)
        denominator = np.full((nstates, nmatrices), self.__log_zero)

        for k in range(0, len(g_matrices)):
            g = g_matrices[k]
            z = z_matrices[k]
            for i in range(0, nstates):
                denominator[i, k] = self.__sum_log_probabilities(g[i, 1:])
                for j in range(0, nstates):
                    numerator[i, j, k] = self.__sum_log_probabilities(z[i, j, 1:])

        numerator2 = np.full((nstates, nstates), self.__log_zero)
        denominator2 = np.full(nstates, self.__log_zero)

        for i in range(0, nstates):
            denominator2[i] = self.__sum_log_probabilities(denominator[i, :])
            for j in range(0, nstates):
                numerator2[i, j] = self.__sum_log_probabilities(numerator[i, j, :])

        transition_matrix = numerator2 - denominator2.reshape((len(denominator2), 1))
        transition_matrix = transition_matrix - self.__sum_log_probability_matrix(np.transpose(transition_matrix)).reshape((nstates, 1))
        
        return transition_matrix

    def __compute_new_variance_matrix(self, feature_matrices, new_mean_matrix, g_matrices):
        nstates = g_matrices[0].shape[0]
        nfeatures = feature_matrices[0].shape[0]
        variance_matrix = np.zeros((nfeatures, nstates), dtype = np.float)
        numerator = np.zeros(nfeatures)
        denominator = 0

        for q in range(0, nstates):
            for i in range(0, len(feature_matrices)):
                feature_matrix = feature_matrices[i]
                g = g_matrices[i]
                numerator = numerator + np.sum( \
                    np.multiply( \
                        np.square( \
                            feature_matrix - new_mean_matrix[:, q].reshape((nfeatures, 1)) \
                        ), \
                        np.exp(g[q, :]) \
                    ), \
                    axis = 1 \
                )
                denominator = denominator + np.sum(np.exp(g[q, :]))
            variance_matrix[:, q] = numerator / float(denominator)
        
        return variance_matrix

    def __convert_vector_to_matrix(self, vector, ncolumns):
        return np.transpose(np.tile(vector, (ncolumns, 1)))

    def __create_plots(self):
        self.__fig, axes = plt.subplots(5, 1)

        axes[0].set_title("Alpha matrix")
        axes[0].xaxis.grid(True)
        axes[0].yaxis.grid(True)
        axes[0].set_xlabel("frames")
        axes[0].grid(linewidth = 3)
        self.__a_plot = axes[0]
        axes[0].imshow(self.__a, aspect='auto')

        axes[1].set_title("Beta matrix")
        axes[1].xaxis.grid(True)
        axes[1].yaxis.grid(True)
        axes[1].set_xlabel("frames")
        axes[1].grid(linewidth = 3)
        self.__b_plot= axes[1]
        axes[1].imshow(self.__b, aspect='auto')
        
        axes[2].set_title("Gamma matrix")
        axes[2].xaxis.grid(True)
        axes[2].yaxis.grid(True)
        axes[2].set_xlabel("frames")
        axes[2].grid(linewidth = 3)
        self.__g_plot= axes[2]
        axes[2].imshow(self.__g, aspect='auto')

        axes[3].set_title("Gamma sum vector")
        axes[3].xaxis.grid(True)
        axes[3].yaxis.grid(True)
        axes[3].set_xlabel("frames")
        axes[3].grid(linewidth = 3)
        self.__g_sum_plot= axes[3]
        axes[3].imshow([np.sum(np.exp(self.__g), axis = 0)], aspect='auto')

        axes[4].set_title("Alpha/Beta sum vector")
        axes[4].xaxis.grid(True)
        axes[4].yaxis.grid(True)
        axes[4].set_xlabel("frames")
        axes[4].grid(linewidth = 3)
        self.__ab_product_sum_plot = axes[4]
        axes[4].imshow([self.__sum_log_probability_matrix(self.__a + self.__b)], aspect='auto')
        
        self.__fig.tight_layout(pad = 0)
        
    def __initialize_hmm_parameters(self, nstates, feature_matrices):
        nfeatures = feature_matrices[0].shape[0]
        initial_state_vector = np.zeros(nstates, dtype = np.float)
        variance_vector = np.zeros(nfeatures, dtype = np.float)
        mean_vector = np.zeros(nfeatures, dtype = np.float)
        transition_matrix = np.full((nstates, nstates), self.__log_zero, dtype = np.float)

        for feature_matrix in feature_matrices:
            variance_vector = np.add(variance_vector, np.var(feature_matrix, axis = 1))
            mean_vector = np.add(mean_vector, np.mean(feature_matrix, axis = 1))    

        variance_vector = np.true_divide(variance_vector, len(feature_matrices))
        mean_vector = np.true_divide(mean_vector, len(feature_matrices))

        variance_matrix = self.__convert_vector_to_matrix(variance_vector, nstates)
        mean_matrix = self.__convert_vector_to_matrix(mean_vector, nstates)

        for j in range(0, nstates):
            mean_variance = np.var(mean_matrix[:, j])
            for i in range(0, nfeatures):
                mean_matrix[i, j] = np.abs(mean_matrix[i, j] + np.random.normal(0, np.sqrt(mean_variance), 1))
                variance_matrix[i, j] = np.abs(variance_matrix[i, j] + np.random.normal(0, np.sqrt(variance_matrix[i, j]), 1))

        for i in range(0, nstates - 1):
            stay_probability = np.random.uniform(0, 1, 1)
            transition_probability = 1 - stay_probability
            transition_matrix[i, i] = np.log(stay_probability)
            transition_matrix[i, i + 1] = np.log(transition_probability)

        # At the end the probability of staying is 100% since it is the end of the HMM
        transition_matrix[-1, -1] = np.log(1)

        for i in range(0, nstates):
            initial_state_vector[i] = np.power(0.5, (i + 1) ** 2)     
        initial_state_vector[0] = initial_state_vector[0] + (1 - np.sum(initial_state_vector))
        initial_state_vector = np.log(initial_state_vector)

        return HMM_Parameters(nstates, initial_state_vector, transition_matrix, mean_matrix, variance_matrix, self.__log_zero)

    def __sum_log_probabilities(self, p):
        a = -np.sort(-np.array(p))
        return a[0] + np.log(1 + np.sum(np.exp(a[1:] - a[0])))

    def __sum_log_probability_vectors(self, p1, p2):
        results = np.full(len(p1), self.__log_zero)
        for i in range(0, len(p1)):
            results[i] = self.__sum_log_probabilities([p1[i], p2[i]])
        return results

    def __sum_log_probability_matrix(self, m):
        width = m.shape[1]
        results = np.full(width, self.__log_zero)
        for i in range(0, m.shape[1]):
            results[i] = self.__sum_log_probabilities(m[:, i])
        return results

    def __train_hmm(self, feature_matrices, nstates, result_queue):
        threshold = 0.05
        old_hmm_parameters = self.__initialize_hmm_parameters(nstates, feature_matrices)
        delta = 1.0

        while delta > threshold:
            new_hmm_parameters = self.__compute_new_hmm_parameters(feature_matrices, old_hmm_parameters)
            old_likelihood = old_hmm_parameters.get_data_log_likelihood()
            new_likelihood = new_hmm_parameters.get_data_log_likelihood()
            delta = np.abs(new_likelihood - old_likelihood) / np.abs(old_likelihood)
            old_hmm_parameters = new_hmm_parameters

            #raw_input("Press any key for next step")
            self.__iteration = self.__iteration + 1

        print("Finished!")
        result_queue.put(new_hmm_parameters)

    def __update_plots(self, frame):
        self.__a_plot.set_title("Alpha matrix (iteration = %d)" % self.__iteration)
        self.__b_plot.set_title("Beta matrix (iteration = %d)" % self.__iteration)
        self.__g_plot.set_title("Gamma matrix (iteration = %d)" % self.__iteration)
        self.__g_sum_plot.set_title("Gamma sum vector (iteration = %d)" % self.__iteration)
        self.__ab_product_sum_plot.set_title("Alpha/beta sum vector (iteration = %d)" % self.__iteration)

        self.__a_plot.imshow(self.__a, aspect='auto')
        self.__b_plot.imshow(self.__b, aspect='auto')
        self.__g_plot.imshow(np.exp(self.__g), aspect='auto')
        self.__g_sum_plot.imshow([np.sum(np.exp(self.__g), axis = 0)], aspect='auto')
        self.__ab_product_sum_plot.imshow([self.__sum_log_probability_matrix(self.__a + self.__b)], aspect='auto')

        return None

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
        self.__a = np.full((nstates, feature_matrices[0].shape[1]), self.__log_zero)
        self.__b = np.full(self.__a.shape, self.__log_zero)
        self.__g = np.full(self.__a.shape, self.__log_zero)
        self.__iteration = 0

        self.__create_plots()
        self.__animation = animation.FuncAnimation(self.__fig, self.__update_plots, interval = 1000, blit = False, repeat = False)

        result = Queue.Queue()
        training_thread = Thread(target = self.__train_hmm, args = [feature_matrices, nstates, result])
        training_thread.start()

        plt.show()
        
        training_thread.join()
        return result.get()

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

class HMM_Parameters:

    def __init__(self, nstates, initial_state_vector, transition_matrix, mean_matrix, variance_matrix, data_log_likelihood):
        self.__data_log_likelihood = data_log_likelihood
        self.__initial_state_vector = initial_state_vector
        self.__mean_matrix = mean_matrix
        self.__nstates = nstates
        self.__transition_matrix = transition_matrix
        self.__variance_matrix = variance_matrix

    def get_data_log_likelihood(self):
        return self.__data_log_likelihood

    def get_initial_state_vector(self):
        return self.__initial_state_vector

    def get_mean_matrix(self):
        return self.__mean_matrix

    def get_nstates(self):
        return self.__nstates
    
    def get_transition_matrix(self):
        return self.__transition_matrix

    def get_variance_matrix(self):
        return self.__variance_matrix

if __name__ == '__main__':
    folder_path = "C:\Users\AkramAsylum\OneDrive\Courses\School\EE 516 - Compute Speech Processing\Assignments\Assignment 5\samples\odessa"

    em = EM()
    result = em.build_hmm_from_folder(folder_path, 10)
    print("Done!")