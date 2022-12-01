import numpy as np
import random
# Fix the random seed to make results reproducible
np.random.seed(42)

import os
import glob
from functools import reduce

def time_shift_signal(wave):
    """ Shift a wave in the time-domain at random
    """
    size = wave.shape[0]
    shift_at_time = np.random.randint(0, size)
    return np.roll(wave, shift_at_time)

def time_shift_spectrogram(spectrogram):
    """ Shift a spectrogram along the time axis in the spectral-domain at random
    """
    nb_cols = spectrogram.shape[1]
    nb_shifts = np.random.randint(0, nb_cols)

    return np.roll(spectrogram, nb_shifts, axis=1)

def pitch_shift_spectrogram(spectrogram):
    """ Shift a spectrogram along the frequency axis in the spectral-domain at
    random
    """
    nb_cols = spectrogram.shape[0]
    max_shifts = nb_cols//20 # around 5% shift
    nb_shifts = np.random.randint(-max_shifts, max_shifts)

    return np.roll(spectrogram, nb_shifts, axis=0)

def same_class_augmentation(wave, class_dir):
    """ Perform same class augmentation of the wave by loading a random segment
    from the class_dir and additively combine the wave with that segment.
    """
    sig_paths = glob.glob(os.path.join(class_dir, "*.wav"))
    aug_sig_path = random.choice(sig_paths)
    (fs, aug_sig) = read_wave_file(aug_sig_path)
    alpha = np.random.rand()
    wave = (1.0-alpha)*wave + alpha*aug_sig
    return wave