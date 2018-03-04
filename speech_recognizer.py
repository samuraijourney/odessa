from asr_feature_builder import ASR_Feature_Builder
from Queue import LifoQueue
from speech_sampler import Speech_Sampler
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

    def __init__(self):

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
        self.__stop_processing = False

        self.__feature_builder.set_plot_blocking(True)
        self.__sampler.add_sample_callback(self.__queue_speech_segment)
        self.__sampler.hide_spectrogram_plot()

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
                            "   3) Plot full feature matrix\n"
                            "   4) Plot mfcc feature matrix\n"
                            "   5) Plot delta feature matrix\n"
                            "   6) Plot filter banks\n"
                            "   7) Plot filter bank filtered spectra sum\n"
                            "   8) Plot filter bank filtered spectra sum log\n"
                            "   9) Plot filter bank filtered spectra sum log dct (mfcc)\n"
                            "\n"
                            "To resume you have to exit the plot cause matplotlib is stupid...\n"
                        )
            
                print message
                text = raw_input("Enter your selection: ")

                try: 
                    option = int(text)
                    if option > 9:
                        continue
                    else:
                        invalid_selection = False

                    if option == 0:
                        self.__sampler.pause()
                    elif option == 1:
                        self.__plot_option = -1
                    elif option == 2:
                        self.__empty_speech_segment_queue()
                    else:
                        self.__plot_option = option - 3
                                   
                except ValueError:
                    continue

            time.sleep(0.1)

    def __process_speech_segments(self):
        while not self.__stop_processing:
            time.sleep(self.__process_sleep_time)
            speech_segment = self.__get_speech_segment()
            if speech_segment == None:
                continue
            
            self.__feature_builder.compute_features_for_signal( \
                speech_segment, \
                self.__fs, \
                self.__feature_nfilters, \
                self.__feature_window_duration, \
                self.__feature_skip_duration, \
                self.__feature_radius, \
                self.__feature_nfilters_keep)

            self.__pool.map(asr_feature_builder_plot, \
                [ASR_Feature_Builder_Plot_Options( \
                    self.__feature_builder, \
                    self.__plot_option, \
                    self.__feature_nfilters_keep \
                )])

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

    def run(self):
        processing_thread = threading.Thread(target = self.__process_speech_segments)
        interactive_thread = threading.Thread(target = self.__handle_interactive)

        processing_thread.start()
        interactive_thread.start()
        self.__sampler.run()

        self.__stop_processing = True
        processing_thread.join()
        interactive_thread.join()

if __name__ == '__main__':
    speech_recognizer = Speech_Recognizer()
    speech_recognizer.run()