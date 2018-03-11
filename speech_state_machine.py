import operator

class Speech_State_Machine:

    def __init__(self):
        self.__callbacks = []
        self.__hmm_phrase_map = {}
        self.__hmm_threshold_map = {}
        self.__primary_hmm = None
        self.__primary_signaled = False

    def __run_callbacks(self, speech_hmm, phrase, log_match_probability, is_primary):
        for callback in self.__callbacks:
            callback(speech_hmm, phrase, log_match_probability, is_primary)

    def set_primary_hmm(self, speech_hmm, phrase, match_threshold):
        self.__hmm_phrase_map[speech_hmm] = phrase
        self.__hmm_threshold_map[speech_hmm] = match_threshold
        self.__primary_hmm = speech_hmm

    def add_secondary_hmm(self, speech_hmm, phrase, match_threshold):
        self.__hmm_phrase_map[speech_hmm] = phrase
        self.__hmm_threshold_map[speech_hmm] = match_threshold

    def add_speech_match_callback(self, callback):
        self.__callbacks.append(callback)

    def update(self, feature_matrix):
        if self.__primary_hmm is None:
            return

        if (not self.__primary_signaled):
            match = self.__primary_hmm.match_from_feature_matrix(feature_matrix)
            if match > self.__hmm_threshold_map[self.__primary_hmm]:
                self.__primary_signaled = True
                self.__run_callbacks(self.__primary_hmm, self.__hmm_phrase_map[self.__primary_hmm], match, True)
        else:
            hmm_probability_map = {}

            for speech_hmm in self.__hmm_phrase_map:
                hmm_probability_map[speech_hmm] = speech_hmm.match_from_feature_matrix(feature_matrix)
            selected_hmm, match = max(hmm_probability_map.iteritems(), key = operator.itemgetter(1))

            if match > self.__hmm_threshold_map[selected_hmm]:
                self.__run_callbacks(selected_hmm, self.__hmm_phrase_map[selected_hmm], match, False)
                self.__primary_signaled = False

        print("Match: %.3f" % match)