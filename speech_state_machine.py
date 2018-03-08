import operator

class Speech_State_Machine:

    def __init__(self):
        self.__callbacks = []
        self.__hmm_phrase_map = {}
        self.__log_match_threshold = -10e10
        self.__primary_hmm = None
        self.__primary_signaled = False

    def __run_callbacks(self, speech_hmm, phrase, is_primary):
        for callback in self.__callbacks:
            callback(speech_hmm, phrase, is_primary)

    def set_primary_hmm(self, hmm, phrase):
        self.__hmm_phrase_map[hmm] = phrase
        self.__primary_hmm = hmm

    def add_secondary_hmm(self, speech_hmm, phrase):
        self.__hmm_phrase_map[speech_hmm] = phrase

    def add_speech_match_callback(self, callback):
        self.__callbacks.append(callback)

    def update(self, feature_matrix):
        if self.__primary_hmm is None:
            return

        if (not self.__primary_signaled):
            if self.__primary_hmm.match(feature_matrix, self.__log_match_threshold):
                self.__primary_signaled = True
                self.__run_callbacks(self.__primary_hmm, self.__hmm_phrase_map[self.__primary_hmm], True)
        else:
            hmm_probability_map = {}

            for speech_hmm in self.__hmm_phrase_map:
                hmm_probability_map[speech_hmm] = speech_hmm.match_probability(feature_matrix)
            selected_hmm, log_probability = max(hmm_probability_map.iteritems(), key = operator.itemgetter(1))

            if log_probability > self.__log_match_threshold:
                self.__run_callbacks(selected_hmm, self.__hmm_phrase_map[selected_hmm], False)
                self.__primary_signaled = False