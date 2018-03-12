from asr_feature_builder import ASR_Feature_Builder
from matplotlib import animation
from threading import Thread
import hmm
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

    def __compute_ab_matrices(self, feature_matrix, hmm_parameters):
        mean_matrix = hmm_parameters.get_mean_matrix()
        nstates = hmm_parameters.get_nstates()
        transition_matrix = hmm_parameters.get_transition_matrix()
        variance_matrix = hmm_parameters.get_variance_matrix()

        a = np.full((nstates, feature_matrix.shape[1]), self.__log_zero)
        b = np.full(a.shape, self.__log_zero)

        a[0, 0] = np.log(1)
        b[-1, -1] = np.log(1)

        for t in range(1, a.shape[1]):
            p_a = self.__compute_gaussian_probability_log(feature_matrix[:, t], mean_matrix, variance_matrix)
            p_b = self.__compute_gaussian_probability_log(feature_matrix[:, -t], mean_matrix, variance_matrix)
            for j in range(0, nstates):
                a[j, t] = self.__sum_log_probabilities(a[:, t - 1] + transition_matrix[:, j]) + p_a[j]
                b[j, -t - 1] = self.__sum_log_probabilities(b[:, -t] + transition_matrix[j, :] + p_b)

        return a, b

    def __compute_data_log_likelihood(self, a_matrices):
        p = 0
        for i in range(0, len(a_matrices)):
            p = p + a_matrices[i][-1, -1]
        return p

    def __compute_gaussian_probability_log(self, feature_vector, mean_matrix, variance_matrix):
        nstates = mean_matrix.shape[1]
        nfeatures = len(feature_vector)
        feature_matrix = self.__convert_vector_to_matrix(feature_vector, nstates)
        exponent = -0.5 * np.sum(np.true_divide(np.square(feature_matrix - mean_matrix), variance_matrix), axis = 0)
        denominator = 0.5 * nfeatures * np.log(2 * np.pi) + 0.5 * np.sum(np.log(variance_matrix), axis = 0)
        return exponent - denominator

    def __compute_gz_matrices(self, a, b, feature_matrix, hmm_parameters):
        mean_matrix = hmm_parameters.get_mean_matrix()
        transition_matrix = hmm_parameters.get_transition_matrix()
        variance_matrix = hmm_parameters.get_variance_matrix()
        nstates = a.shape[0]
        nframes = a.shape[1]

        g = np.full(a.shape, self.__log_zero)
        z = np.full((nstates, nstates, nframes), self.__log_zero)

        g[:, 0] = a[:, 0] + b[:, 0] - self.__sum_log_probabilities(a[:, 0] + b[:, 0])
        z[-1, -1, -1] = np.log(1)

        for t in range(1, nframes):
            g[:, t] = a[:, t] + b[:, t] - self.__sum_log_probabilities(a[:, t] + b[:, t])
            p = self.__compute_gaussian_probability_log(feature_matrix[:, t], mean_matrix, variance_matrix)
            for i in range(0, nstates):
                z[i, i, t - 1] = b[i, t] + a[i, t - 1] + transition_matrix[i, i] + p[i] - a[-1, -1]
                if i != 0:
                    z[i - 1, i, t - 1] = b[i, t] + a[i - 1, t - 1] + transition_matrix[i - 1, i] + p[i] - a[-1, -1]

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
        new_mean_matrix, new_variance_matrix = self.__compute_new_observation(feature_matrices, g_matrices)
        data_log_likelihood = self.__compute_data_log_likelihood(a_matrices)
        
        return hmm.HMM_Parameters(hmm_parameters.get_nstates(), new_initial_state_vector, new_transition_matrix, new_mean_matrix, new_variance_matrix, data_log_likelihood)

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
                g_q = np.exp(g[q, :])
                numerator = numerator + np.sum(np.multiply(feature_matrix, g_q), axis = 1)
                denominator = denominator + np.sum(g_q)
            mean_matrix[:, q] = numerator / float(denominator)
        
        return mean_matrix

    def __compute_new_observation(self, feature_matrices, g_matrices):
        nstates = g_matrices[0].shape[0]
        nfeatures = feature_matrices[0].shape[0]
        mean_matrix = np.zeros((nfeatures, nstates), dtype = np.float)
        variance_matrix = np.zeros((nfeatures, nstates), dtype = np.float)

        for q in range(0, nstates):
            denominator = 0

            for i in range(0, len(feature_matrices)):
                feature_matrix = feature_matrices[i]
                g_matrix = g_matrices[i]
                g_matrix_exp = np.exp(g_matrix)
                denominator = denominator + np.sum(g_matrix_exp[q, :])
                mean_matrix[:, q] = mean_matrix[:, q] + np.sum(np.multiply(feature_matrix, g_matrix_exp[q, :]), axis = 1)

            mean_matrix[:, q] = mean_matrix[:, q] / denominator

            for i in range(0, len(feature_matrices)):
                feature_matrix = feature_matrices[i]
                g_matrix = g_matrices[i]
                g_matrix_exp = np.exp(g_matrix)
                variance_matrix[:, q] = variance_matrix[:, q] + \
                    np.sum( \
                        np.multiply( \
                            np.square( \
                                feature_matrix - mean_matrix[:, q].reshape((nfeatures, 1))
                            ), \
                            np.tile( \
                                g_matrix_exp[q, :], \
                                (nfeatures, 1)
                            ) \
                        ), \
                        axis = 1
                    )
     
            variance_matrix[:, q] = variance_matrix[:, q] / denominator

        return mean_matrix, variance_matrix

    def __compute_new_state_transition_matrix(self, g_matrices, z_matrices):
        nstates = g_matrices[0].shape[0]
        nmatrices = len(g_matrices)
        numerator = np.full((nstates, nstates, nmatrices), self.__log_zero)
        denominator = np.full((nstates, nmatrices), self.__log_zero)

        for k in range(0, nmatrices):
            g = g_matrices[k]
            z = z_matrices[k]
            for i in range(0, nstates):
                denominator[i, k] = self.__sum_log_probabilities(g[i, :])
                for j in range(0, nstates):
                    numerator[i, j, k] = self.__sum_log_probabilities(z[i, j, :])

        numerator2 = np.full((nstates, nstates), self.__log_zero)
        denominator2 = np.full(nstates, self.__log_zero)

        for i in range(0, nstates):
            denominator2[i] = self.__sum_log_probabilities(denominator[i, :])
            for j in range(0, nstates):
                numerator2[i, j] = self.__sum_log_probabilities(numerator[i, j, :])

        return numerator2 - denominator2.reshape((len(denominator2), 1))

    def __compute_new_variance_matrix(self, feature_matrices, new_mean_matrix, g_matrices):
        nstates = g_matrices[0].shape[0]
        nfeatures = feature_matrices[0].shape[0]
        variance_matrix = np.zeros((nfeatures, nstates), dtype = np.float)
        numerator = np.zeros(nfeatures, dtype = np.float)
        denominator = 0

        for q in range(0, nstates):
            for i in range(0, len(feature_matrices)):
                feature_matrix = feature_matrices[i]
                g = g_matrices[i]
                g_q = np.exp(g[q, :])
                numerator = numerator + np.sum( \
                    np.multiply( \
                        np.square( \
                            feature_matrix - new_mean_matrix[:, q].reshape((nfeatures, 1)) \
                        ), \
                        g_q \
                    ), \
                    axis = 1 \
                )
                denominator = denominator + np.sum(g_q)
            variance_matrix[:, q] = numerator / float(denominator)
        
        return variance_matrix

    def __convert_vector_to_matrix(self, vector, ncolumns):
        return np.transpose(np.tile(vector, (ncolumns, 1)))

    def __create_plots(self):
        self.__fig, axes = plt.subplots(3, 1)

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
        
        self.__fig.tight_layout(pad = 0)
        
    def __initialize_hmm_parameters(self, nstates, feature_matrices):
        nfeatures = feature_matrices[0].shape[0]
        initial_state_vector = np.zeros(nstates, dtype = np.float)
        variance_vector = np.zeros(nfeatures, dtype = np.float)
        mean_vector = np.zeros(nfeatures, dtype = np.float)
        transition_matrix = np.full((nstates, nstates), self.__log_zero, dtype = np.float)

        for feature_matrix in feature_matrices:
            variance_vector = variance_vector + np.var(feature_matrix, axis = 1)
            mean_vector = mean_vector + np.mean(feature_matrix, axis = 1)

        variance_vector = np.true_divide(variance_vector, len(feature_matrices))
        mean_vector = np.true_divide(mean_vector, len(feature_matrices))

        variance_matrix = self.__convert_vector_to_matrix(variance_vector, nstates)
        mean_matrix = self.__convert_vector_to_matrix(mean_vector, nstates)

        for i in range(0, nfeatures):
            mean_std = np.sqrt(variance_vector[i])
            for j in range(0, nstates):
                mean_matrix[i, j] = mean_matrix[i, j] + np.random.normal(0, mean_std, 1)

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

        return hmm.HMM_Parameters(nstates, initial_state_vector, transition_matrix, mean_matrix, variance_matrix, self.__log_zero)

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

    def __train_hmm(self, feature_matrices, nstates, result_queue, max_iterations, convergence_threshold):
        old_hmm_parameters = self.__initialize_hmm_parameters(nstates, feature_matrices)
        delta = np.inf
        new_likelihood = 0.0

        while (delta > convergence_threshold) and (self.__iteration < max_iterations):
            new_hmm_parameters = self.__compute_new_hmm_parameters(feature_matrices, old_hmm_parameters)
            old_likelihood = old_hmm_parameters.get_data_log_likelihood()
            new_likelihood = new_hmm_parameters.get_data_log_likelihood()
            delta = np.abs(new_likelihood - old_likelihood)
            old_hmm_parameters = new_hmm_parameters

            print("\tIteration %d - Delta: %.8f, Likelihood: %.4f" % (self.__iteration, delta, new_likelihood))
            self.__iteration = self.__iteration + 1

        print("Finished - Delta: %.8f, Likelihood: %.4f" % (delta, new_likelihood))
        result_queue.put(new_hmm_parameters)

    def __update_plots(self, frame):
        self.__a_plot.set_title("Alpha matrix (iteration = %d)" % self.__iteration)
        self.__b_plot.set_title("Beta matrix (iteration = %d)" % self.__iteration)
        self.__g_plot.set_title("Gamma matrix (iteration = %d)" % self.__iteration)

        self.__a_plot.imshow(self.__a, aspect='auto')
        self.__b_plot.imshow(self.__b, aspect='auto')
        self.__g_plot.imshow(np.exp(self.__g), aspect='auto')

        return None

    def build_hmm_from_folder(self, folder_path, nstates, max_iterations = 200, convergence_threshold = 0.001, show_plots = False):
        audio_files = []

        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                audio_files.append(file_path)
        
        return self.build_hmm_from_files(audio_files, nstates, max_iterations, convergence_threshold, show_plots)

    def build_hmm_from_files(self, audio_files, nstates, max_iterations = 200, convergence_threshold = 0.001, show_plots = False):
        signals = []
        fs = -1

        for audio_file in audio_files:
            fs, s = wavfile.read(audio_file, 'rb')
            signals.append(s)
        
        return self.build_hmm_from_signals(signals, fs, nstates, max_iterations, convergence_threshold, show_plots)

    def build_hmm_from_feature_matrices(self, feature_matrices, nstates, max_iterations = 200, convergence_threshold = 0.001, show_plots = False):
        self.__a = np.full((nstates, feature_matrices[0].shape[1]), self.__log_zero)
        self.__b = np.full(self.__a.shape, self.__log_zero)
        self.__g = np.full(self.__a.shape, self.__log_zero)
        self.__iteration = 0

        if show_plots:
            self.__create_plots()
            self.__animation = animation.FuncAnimation(self.__fig, self.__update_plots, interval = 1000, blit = False, repeat = False)

        result = Queue.Queue()
        training_thread = Thread(target = self.__train_hmm, args = [feature_matrices, nstates, result, max_iterations, convergence_threshold])
        training_thread.start()

        if show_plots:
            plt.show()
        
        training_thread.join()

        new_hmm = hmm.HMM()
        new_hmm.initialize_from_hmm_parameters(result.get())

        return new_hmm

    def build_hmm_from_signals(self, signals, fs, nstates, max_iterations = 200, convergence_threshold = 0.001, show_plots = False):
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
        
        return self.build_hmm_from_feature_matrices(feature_matrices, nstates, max_iterations, convergence_threshold, show_plots)