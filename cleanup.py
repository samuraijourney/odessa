from optparse import OptionParser
import msvcrt
import os
import scipy.io.wavfile as wavfile
import sounddevice as sd
import sys

parser = OptionParser()
parser.add_option("-f", "--folderpath", dest="folderpath", help="Folder path of .wav files to clean")
(options, args) = parser.parse_args()

if not options.folderpath:
    parser.error('folderpath of .wav recordings')
    sys.exit(-1)

raw_input("Press enter to start cleanup")
deletion_list = []
for file in os.listdir(options.folderpath):
    if file.endswith(".wav"):
        file_path = os.path.join(options.folderpath, file)
        fs, s = wavfile.read(file_path, 'rb')
        sd.play(s, fs, blocking = True)
        print("Press enter to keep, space to delete: %s" % file)
        input_char = msvcrt.getch()
        if input_char.upper() == ' ': 
            deletion_list.append(file_path)
            print("x")
        else:
            print("")

for file_path in deletion_list:
    os.remove(file_path)
    print("Deleting %s" % file_path)