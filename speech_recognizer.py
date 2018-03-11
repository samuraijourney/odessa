from asr_feature_builder import ASR_Feature_Builder
from Queue import LifoQueue
from speech_sampler import Speech_Sampler
from speech_state_machine import Speech_State_Machine
import hmm
import multiprocessing
import numpy as np
import os
import threading
import time

np.warnings.filterwarnings('ignore')

def asr_feature_builder_plot(plot_options):
    feature_builder = plot_options.get_feature_builder()
    nfilters_to_keep = plot_options.get_nfilters_to_keep()
    option = plot_options.get_option()

    if option == 0:
        feature_builder.plot_features_matrix()
    elif option == 1:
        feature_builder.plot_mfcc_features_matrix(nfilters_to_keep)
    elif option == 2:
        feature_builder.plot_delta_features_matrix(nfilters_to_keep)
    elif option == 3:
        feature_builder.plot_all_filter_banks()
    elif option == 4:
        feature_builder.plot_all_filtered_spectra_sum()
    elif option == 5:
        feature_builder.plot_all_filtered_spectra_sum_log()
    elif option == 6:
        feature_builder.plot_all_filtered_spectra_mfcc()
    elif option == 7:
        feature_builder.plot_mfcc_transitions(nfilters_to_keep)
    elif option == 8:
        feature_builder.plot_signal()

class ASR_Feature_Builder_Plot_Options:

        def __init__(self, feature_builder, option, nfilters_to_keep):
            self.__feature_builder = feature_builder
            self.__nfilters_to_keep = nfilters_to_keep
            self.__option = option
        
        def get_feature_builder(self):
            return self.__feature_builder

        def get_nfilters_to_keep(self):
            return self.__nfilters_to_keep

        def get_option(self):
            return self.__option

class Speech_Recognizer:

    def __init__(self, speech_state_machine):

        # Speech sampler
        self.__fs = 16000
        self.__sampler_window_duration = 5 # seconds
        self.__sampler = Speech_Sampler(self.__sampler_window_duration, self.__fs)

        # Feature builder
        self.__feature_window_duration = 0.025 # seconds
        self.__feature_skip_duration = 0.01 # seconds
        self.__feature_nfilters = 26
        self.__feature_nfilters_keep = 13
        self.__feature_radius = 2
        self.__feature_builder = ASR_Feature_Builder()

        # Processing
        self.__max_queue_size = 10
        self.__pool = multiprocessing.Pool()
        self.__process_sleep_time = 0.025 # seconds
        self.__queue_lock = threading.Lock()
        self.__plot_option = -1
        self.__speech_segments = LifoQueue()
        self.__speech_state_machine = speech_state_machine
        self.__stop_processing = False

        self.__feature_builder.set_plot_blocking(True)
        self.__sampler.add_sample_callback(self.__queue_speech_segment)
        self.__sampler.hide_spectrogram_plot()
        self.__speech_state_machine.add_speech_match_callback(self.__speech_matched)

    def __empty_speech_segment_queue(self):
        self.__queue_lock.acquire(True)
        while not self.__speech_segments.empty():
            self.__speech_segments.get()
        self.__queue_lock.release()

    def __get_speech_segment(self):
        speech_segment = None
        self.__queue_lock.acquire(True)
        if not self.__speech_segments.empty():
            speech_segment = self.__speech_segments.get()
        self.__queue_lock.release()
        return speech_segment

    def __handle_interactive(self):
        while not self.__stop_processing:
            invalid_selection = True
            while invalid_selection:
                os.system('cls')
                message = ( "Please enter the number of the option you wish to execute:\n"
                            "   0) Pause\n"
                            "   1) No plots\n"
                            "   2) Clear data queue\n"
                            "   3) Start save speech segments (saves in current working directory)\n"
                            "   4) Stop save speech segments\n"
                            "   5) Start speech segment playback\n"
                            "   6) Stop speech segment playback\n"
                            "   7) Plot full feature matrix\n"
                            "   8) Plot mfcc feature matrix\n"
                            "   9) Plot delta feature matrix\n"
                            "  10) Plot filter banks\n"
                            "  11) Plot filter bank filtered spectra sum\n"
                            "  12) Plot filter bank filtered spectra sum log\n"
                            "  13) Plot filter bank filtered spectra sum log dct (mfcc)\n"
                            "  14) Plot mfcc transitions\n"
                            "  15) Plot speech segment\n"
                            "\n"
                            "To resume you have to exit the plot cause matplotlib is stupid...\n"
                        )
            
                print message
                text = raw_input("Enter your selection: ")

                try: 
                    option = int(text)
                    if option > 15:
                        continue
                    else:
                        invalid_selection = False

                    if option == 0:
                        self.__sampler.pause()
                    elif option == 1:
                        self.__plot_option = -1
                    elif option == 2:
                        self.__empty_speech_segment_queue()
                    elif option == 3:
                        self.__sampler.save_samples(os.getcwd())
                    elif option == 4:
                        self.__sampler.save_samples(None, False)
                    elif option == 5:
                        self.__sampler.play_samples(True)
                    elif option == 6:
                        self.__sampler.play_samples(False)
                    else:
                        self.__plot_option = option - 7
                                   
                except ValueError:
                    continue

            time.sleep(0.1)

    def __process_speech_segments(self):
        while not self.__stop_processing:
            time.sleep(self.__process_sleep_time)
            speech_segment = self.__get_speech_segment()
            if speech_segment is None:
                continue

            feature_matrix = self.__feature_builder.compute_features_for_signal( \
                speech_segment * np.iinfo(np.int16).max, \
                self.__fs, \
                self.__feature_nfilters, \
                self.__feature_window_duration, \
                self.__feature_skip_duration, \
                self.__feature_radius, \
                self.__feature_nfilters_keep)

            self.__speech_state_machine.update(feature_matrix)

            plot_options = ASR_Feature_Builder_Plot_Options( \
                    self.__feature_builder, \
                    self.__plot_option, \
                    self.__feature_nfilters_keep \
                )

            self.__pool.map(asr_feature_builder_plot, [plot_options])

    def __queue_speech_segment(self, speech_segment):
        self.__queue_lock.acquire(True)
        if self.__speech_segments.qsize() == self.__max_queue_size:
            temp_queue = LifoQueue()
            while not self.__speech_segments.empty():
                temp_queue.put(self.__speech_segments.get())
            
            # Discard the oldest data
            temp_queue.get()
            
            while not temp_queue.empty():
                self.__speech_segments.put(temp_queue.get())
        self.__speech_segments.put(speech_segment)
        self.__queue_lock.release()

    def __speech_matched(self, hmm, phrase, log_match_probability, is_primary):
        print(phrase)

    def run(self, interactive = True):
        processing_thread = threading.Thread(target = self.__process_speech_segments)
        if interactive:
            interactive_thread = threading.Thread(target = self.__handle_interactive)

        processing_thread.start()
        if interactive:
            interactive_thread.start()
        self.__sampler.run()

        self.__stop_processing = True
        processing_thread.join()
        if interactive:
            interactive_thread.join()

if __name__ == '__main__':
    hmm_folder_path = "C:\Users\AkramAsylum\OneDrive\Courses\School\EE 516 - Compute Speech Processing\Assignments\Assignment 5\hmm"

    odessa_hmm = os.path.join(hmm_folder_path, "odessa.hmm")
    play_music_hmm = os.path.join(hmm_folder_path, "play_music.hmm")
    stop_music_hmm = os.path.join(hmm_folder_path, "stop_music.hmm")
    turn_on_the_lights_hmm = os.path.join(hmm_folder_path, "turn_on_the_lights.hmm")
    turn_off_the_lights_hmm = os.path.join(hmm_folder_path, "turn_off_the_lights.hmm")
    what_time_is_it_hmm = os.path.join(hmm_folder_path, "what_time_is_it.hmm")

    odessa_threshold = -900.0
    play_music_threshold = -900.0
    stop_music_threshold = -1100.0
    turn_on_the_lights_threshold = -1100.0
    turn_off_the_lights_threshold = -1100.0
    what_time_is_it_threshold = -1000.0

    speech_state_machine = Speech_State_Machine()
    if os.path.exists(odessa_hmm):
        odessa_speech_hmm = hmm.HMM()
        odessa_speech_hmm.initialize_from_file(odessa_hmm)
        speech_state_machine.set_primary_hmm(odessa_speech_hmm, "odessa", odessa_threshold)
    if os.path.exists(play_music_hmm):
        play_music_speech_hmm = hmm.HMM()
        play_music_speech_hmm.initialize_from_file(play_music_hmm)
        speech_state_machine.add_secondary_hmm(play_music_speech_hmm, "play music", play_music_threshold)
    if os.path.exists(stop_music_hmm):
        stop_music_speech_hmm = hmm.HMM()
        stop_music_speech_hmm.initialize_from_file(stop_music_hmm)
        speech_state_machine.add_secondary_hmm(stop_music_speech_hmm, "stop music", stop_music_threshold)
    if os.path.exists(turn_on_the_lights_hmm):
        turn_on_the_lights_speech_hmm = hmm.HMM()
        turn_on_the_lights_speech_hmm.initialize_from_file(turn_on_the_lights_hmm)
        speech_state_machine.add_secondary_hmm(turn_on_the_lights_speech_hmm, "turn on the lights", turn_on_the_lights_threshold)
    if os.path.exists(turn_off_the_lights_hmm):
        turn_off_the_lights_speech_hmm = hmm.HMM()
        turn_off_the_lights_speech_hmm.initialize_from_file(turn_off_the_lights_hmm)
        speech_state_machine.add_secondary_hmm(turn_off_the_lights_speech_hmm, "turn off the lights", turn_off_the_lights_threshold)
    if os.path.exists(what_time_is_it_hmm):
        what_time_is_it_speech_hmm = hmm.HMM()
        what_time_is_it_speech_hmm.initialize_from_file(what_time_is_it_hmm)
        speech_state_machine.add_secondary_hmm(what_time_is_it_speech_hmm, "what time is it", what_time_is_it_threshold)

    speech_recognizer = Speech_Recognizer(speech_state_machine)
    speech_recognizer.run(True)