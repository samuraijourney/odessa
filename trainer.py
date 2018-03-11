from em import EM
import os

if __name__ == '__main__':
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

    em = EM()
    for item in training_list:
        folder_name = os.path.split(item[1])[-1]
        print("Training %s..." % folder_name)

        speech_hmm = em.build_hmm_from_folder(item[1], item[3], False)
        speech_hmm.save(item[2])

        matches = speech_hmm.match_from_folder(item[1])
        for i in range(0, len(matches)):
            print("training match: %.3f" % matches[i])

        validation_matches = speech_hmm.match_from_folder(item[0])
        for i in range(0, len(validation_matches)):
            print("validation match: %.3f" % validation_matches[i])

        if garbage_validation_samples != item[1]:
            matches = speech_hmm.match_from_folder(garbage_samples)
            for i in range(0, len(matches)):
                print("garbage match: %.3f" % matches[i])

        if odessa_samples != item[1]:
            matches = speech_hmm.match_from_folder(odessa_samples)
            for i in range(0, len(matches)):
                print("odessa match: %.3f" % matches[i])

        if play_music_samples != item[1]:
            matches = speech_hmm.match_from_folder(play_music_samples)
            for i in range(0, len(matches)):
                print("play music match: %.3f" % matches[i])

        if stop_music_samples != item[1]:
            matches = speech_hmm.match_from_folder(stop_music_samples)
            for i in range(0, len(matches)):
                print("stop music match: %.3f" % matches[i])

        if turn_on_the_lights_samples != item[1]:
            matches = speech_hmm.match_from_folder(turn_on_the_lights_samples)
            for i in range(0, len(matches)):
                print("turn on the lights match: %.3f" % matches[i])

        if turn_off_the_lights_samples != item[1]:
            matches = speech_hmm.match_from_folder(turn_off_the_lights_samples)
            for i in range(0, len(matches)):
                print("turn off the lights match: %.3f" % matches[i])

        if what_time_is_it_samples != item[1]:
            matches = speech_hmm.match_from_folder(what_time_is_it_samples)
            for i in range(0, len(matches)):
                print("what time is it match: %.3f" % matches[i])

        print("")

    print("Training complete")