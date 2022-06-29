# Say hello Odessa!

Odessa is a basic speech recognition system which can identify very specific phrases of my own speech and respond accordingly,
with......spunk.

This algorithmic detection of speech and signal processing were all done from scratch in Python through the use of numpy, scipy, matplotlib, sounddevice and soundfile libraries.

* All audio training samples for the phrases can be found [here](https://github.com/samuraijourney/Odessa/tree/master/samples).
* All generated hmm binaries for the audio samples are found [here](https://github.com/samuraijourney/Odessa/tree/master/hmm).
* The generation of speech features from a sound signal is done via the [asr_feature_builder.py](https://github.com/samuraijourney/Odessa/tree/master/asr_feature_builder.py).
* The training of hmm binaries from audio samples is done via the [em.py](https://github.com/samuraijourney/Odessa/tree/master/em.py).
* The execution of the program funneling identified live speech through recognition is done via the [speech_recognizer.py](https://github.com/samuraijourney/Odessa/tree/master/speech_recognizer.py).
* The detection of live speech segments and sampling to disk is done via the [speech_sampler.py](https://github.com/samuraijourney/Odessa/tree/master/speech_sampler.py).
* The hot word tracking of odessa and subsequent phrase recognitions are done via the [speech_state_machine.py](https://github.com/samuraijourney/Odessa/tree/master/speech_state_machine.py).
* The funneling of audio samples in for training with different HMM parameters looking for an optimal configuration and generating "Training Results.xlsx" are done through [trainer.py](https://github.com/samuraijourney/Odessa/tree/master/trainer.py).

The final application looks like this!

[![Alt text](https://img.youtube.com/vi/Rcp9Yd4NTCE/0.jpg)](https://www.youtube.com/watch?v=Rcp9Yd4NTCE)
