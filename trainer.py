from em import EM
import numpy as np
import os

import sys

class Dual_Printer(object):
    def __init__(self, *files):
        self.__files = files

    def write(self, obj):
        for f in self.__files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately

    def flush(self) :
        for f in self.__files:
            f.flush()

class Csv_Printer(object):
    def __init__(self, *files):
        self.__files = files

    def write(self, data):
        line = ",".join(str(e) for e in data) + "\n"
        for f in self.__files:   
            f.write(line)
            f.flush() # If you want the output to be visible immediately

    def flush(self) :
        for f in self.__files:
            f.flush()

if __name__ == '__main__':
    f1 = open('trainer.txt', 'w')
    f2 = open('results.csv', 'w')
    original = sys.stdout
    sys.stdout = Dual_Printer(sys.stdout, f1)
    csv = Csv_Printer(f2)

    training_list = []

    hmm_folder_path = "C:\Users\AkramAsylum\OneDrive\Courses\School\EE 516 - Compute Speech Processing\Assignments\Assignment 5\hmm"
    hmm_sample_path = "C:\Users\AkramAsylum\OneDrive\Courses\School\EE 516 - Compute Speech Processing\Assignments\Assignment 5\samples"

    odessa_hmm = os.path.join(hmm_folder_path, "odessa.hmm")
    play_music_hmm = os.path.join(hmm_folder_path, "play_music.hmm")
    stop_music_hmm = os.path.join(hmm_folder_path, "stop_music.hmm")
    turn_on_the_lights_hmm = os.path.join(hmm_folder_path, "turn_on_the_lights.hmm")
    turn_off_the_lights_hmm = os.path.join(hmm_folder_path, "turn_off_the_lights.hmm")
    what_time_is_it_hmm = os.path.join(hmm_folder_path, "what_time_is_it.hmm")

    garbage_samples = os.path.join(hmm_sample_path, "garbage")
    odessa_samples = os.path.join(hmm_sample_path, "odessa")
    play_music_samples = os.path.join(hmm_sample_path, "play_music")
    stop_music_samples = os.path.join(hmm_sample_path, "stop_music")
    turn_on_the_lights_samples = os.path.join(hmm_sample_path, "turn_on_the_lights")
    turn_off_the_lights_samples = os.path.join(hmm_sample_path, "turn_off_the_lights")
    what_time_is_it_samples = os.path.join(hmm_sample_path, "what_time_is_it")

    garbage_validation_samples = os.path.join(garbage_samples, "validation")
    odessa_validation_samples = os.path.join(odessa_samples, "validation")
    play_music_validation_samples = os.path.join(play_music_samples, "validation")
    stop_music_validation_samples = os.path.join(stop_music_samples, "validation")
    turn_on_the_lights_validation_samples = os.path.join(turn_on_the_lights_samples, "validation")
    turn_off_the_lights_validation_samples = os.path.join(turn_off_the_lights_samples, "validation")
    what_time_is_it_validation_samples = os.path.join(what_time_is_it_samples, "validation")

    if not os.path.exists(odessa_hmm):
        training_list.append([odessa_validation_samples, odessa_samples, odessa_hmm, 6])
    if not os.path.exists(play_music_hmm):
        training_list.append([play_music_validation_samples, play_music_samples, play_music_hmm, 8])
    if not os.path.exists(stop_music_hmm):
        training_list.append([stop_music_validation_samples, stop_music_samples, stop_music_hmm, 9])
    if not os.path.exists(turn_on_the_lights_hmm):
        training_list.append([turn_on_the_lights_validation_samples, turn_on_the_lights_samples, turn_on_the_lights_hmm, 11])
    if not os.path.exists(turn_off_the_lights_hmm):
        training_list.append([turn_off_the_lights_validation_samples, turn_off_the_lights_samples, turn_off_the_lights_hmm, 11])
    if not os.path.exists(what_time_is_it_hmm):
        training_list.append([what_time_is_it_validation_samples, what_time_is_it_samples, what_time_is_it_hmm, 10])

    csv.write(["Trainee", "# of States", "Iteration", "Test Set", "Max", "Min", "Mean", "Std", "Cutoff"])

    em = EM()
    for item in training_list:
        for q in range(4, 40, 2):
            for iteration in range(0, 3):
                percentile_cutoff = 5
                folder_name = os.path.split(item[1])[-1]
                print("Training %s..." % folder_name)

                speech_hmm = em.build_hmm_from_folder(item[1], q, False)
                #speech_hmm.save(item[2])

                matches = speech_hmm.match_from_folder(item[1])
                for i in range(0, len(matches)):
                    print("\ttraining match: %.3f" % matches[i])
                metrics = [max(matches), min(matches), np.mean(matches), np.std(matches), np.percentile(matches, percentile_cutoff)]
                csv.write([folder_name, q, iteration, "training"] + metrics)
                print("\tmax: %.3f, min: %.3f, mean: %.3f, std: %.3f, cutoff: %.3f" % (metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))
                
                print("---------------------------------------------------------------------")
                
                matches = speech_hmm.match_from_folder(item[0])
                for i in range(0, len(matches)):
                    print("\tvalidation match: %.3f" % matches[i])
                metrics = [max(matches), min(matches), np.mean(matches), np.std(matches), np.percentile(matches, percentile_cutoff)]
                csv.write([folder_name, q, iteration, "validation"] + metrics)
                print("\tmax: %.3f, min: %.3f, mean: %.3f, std: %.3f, cutoff: %.3f" % (metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))

                print("---------------------------------------------------------------------")

                if garbage_validation_samples != item[1]:
                    matches = speech_hmm.match_from_folder(garbage_samples)
                    for i in range(0, len(matches)):
                        print("\tgarbage match: %.3f" % matches[i])
                    metrics = [max(matches), min(matches), np.mean(matches), np.std(matches), np.percentile(matches, percentile_cutoff)]
                    csv.write([folder_name, q, iteration, "garbage"] + metrics)
                    print("\tmax: %.3f, min: %.3f, mean: %.3f, std: %.3f" % (metrics[0], metrics[1], metrics[2], metrics[3]))
                
                print("---------------------------------------------------------------------")

                if odessa_samples != item[1]:
                    matches = speech_hmm.match_from_folder(odessa_samples)
                    for i in range(0, len(matches)):
                        print("\todessa match: %.3f" % matches[i])
                    metrics = [max(matches), min(matches), np.mean(matches), np.std(matches), np.percentile(matches, percentile_cutoff)]
                    csv.write([folder_name, q, iteration, "odessa"] + metrics)
                    print("\tmax: %.3f, min: %.3f, mean: %.3f, std: %.3f" % (metrics[0], metrics[1], metrics[2], metrics[3]))
                
                print("---------------------------------------------------------------------")

                if play_music_samples != item[1]:
                    matches = speech_hmm.match_from_folder(play_music_samples)
                    for i in range(0, len(matches)):
                        print("\tplay music match: %.3f" % matches[i])
                    metrics = [max(matches), min(matches), np.mean(matches), np.std(matches), np.percentile(matches, percentile_cutoff)]
                    csv.write([folder_name, q, iteration, "play music"] + metrics)
                    print("\tmax: %.3f, min: %.3f, mean: %.3f, std: %.3f" % (metrics[0], metrics[1], metrics[2], metrics[3]))
                
                print("---------------------------------------------------------------------")

                if stop_music_samples != item[1]:
                    matches = speech_hmm.match_from_folder(stop_music_samples)
                    for i in range(0, len(matches)):
                        print("\tstop music match: %.3f" % matches[i])
                    metrics = [max(matches), min(matches), np.mean(matches), np.std(matches), np.percentile(matches, percentile_cutoff)]
                    csv.write([folder_name, q, iteration, "stop music"] + metrics)
                    print("\tmax: %.3f, min: %.3f, mean: %.3f, std: %.3f" % (metrics[0], metrics[1], metrics[2], metrics[3]))
                
                print("---------------------------------------------------------------------")

                if turn_on_the_lights_samples != item[1]:
                    matches = speech_hmm.match_from_folder(turn_on_the_lights_samples)
                    for i in range(0, len(matches)):
                        print("\tturn on the lights match: %.3f" % matches[i])
                    metrics = [max(matches), min(matches), np.mean(matches), np.std(matches), np.percentile(matches, percentile_cutoff)]
                    csv.write([folder_name, q, iteration, "turn on the lights"] + metrics)
                    print("\tmax: %.3f, min: %.3f, mean: %.3f, std: %.3f" % (metrics[0], metrics[1], metrics[2], metrics[3]))
                
                print("---------------------------------------------------------------------")

                if turn_off_the_lights_samples != item[1]:
                    matches = speech_hmm.match_from_folder(turn_off_the_lights_samples)
                    for i in range(0, len(matches)):
                        print("\tturn off the lights match: %.3f" % matches[i])
                    metrics = [max(matches), min(matches), np.mean(matches), np.std(matches), np.percentile(matches, percentile_cutoff)]
                    csv.write([folder_name, q, iteration, "turn off the lights"] + metrics)
                    print("\tmax: %.3f, min: %.3f, mean: %.3f, std: %.3f" % (metrics[0], metrics[1], metrics[2], metrics[3]))
                
                print("---------------------------------------------------------------------")

                if what_time_is_it_samples != item[1]:
                    matches = speech_hmm.match_from_folder(what_time_is_it_samples)
                    for i in range(0, len(matches)):
                        print("\twhat time is it match: %.3f" % matches[i])
                    metrics = [max(matches), min(matches), np.mean(matches), np.std(matches), np.percentile(matches, percentile_cutoff)]
                    csv.write([folder_name, q, iteration, "what time is it"] + metrics)
                    print("\tmax: %.3f, min: %.3f, mean: %.3f, std: %.3f" % (metrics[0], metrics[1], metrics[2], metrics[3]))
                
                print("---------------------------------------------------------------------")

                print("")

    print("Training complete")

    sys.stdout = original
    f1.close()
    f2.close()