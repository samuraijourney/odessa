import numpy as np
import pickle

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

class HMM:

    def __compute_gaussian_probability_log(self, feature_vector):
        feature_matrix = np.transpose(np.tile(feature_vector, (self.__nstates, 1)))
        exponent = -0.5 * np.sum(np.true_divide(np.square(feature_matrix - self.__mean_matrix), self.__variance_matrix), axis = 0)
        denominator = -0.5 * self.__nfeatures * np.log(2 * np.pi) - 0.5 * np.sum(np.log(self.__variance_matrix), axis = 0)
        return exponent + denominator

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

    def match(self, feature_matrix, log_match_threshold):
        return self.match_log_probability(feature_matrix) > log_match_threshold

    def match_log_probability(self, feature_matrix):
        match = 0
        current_state = 0

        for t in range(1, feature_matrix.shape[1]):
            p = self.__compute_gaussian_probability_log(feature_matrix[:, t])
            log_probabilities = self.__transition_matrix[current_state, :] + p
            current_state = np.argmax(log_probabilities)
            match = match + log_probabilities[current_state]
        
        return match

    def save(self, file_path):
        file_handle = open(file_path, 'wb')
        pickle.dump(self, file_handle)
        file_handle.close()