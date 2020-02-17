import scipy as sp
import scipy.io.wavfile as wav
import numpy as np
import os
import re
from random import randint
import matplotlib.pyplot as plt
from readwav import read_wav_slices

def preprocess_training_data():
    notes = ['S','C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    regex = "("
    for note in notes[:12]:
        regex += note
        regex += "|"
    regex += notes[12]
    regex += ")[0-9](-[0-100])?.wav$"

    isnotefile = re.compile(regex)
    notefiles = []
    for filename in os.listdir('data/single'):
        match = isnotefile.match(filename)
        if match:
            noteindex = notes.index(match.group(1))
            notefiles.append(['data/single/'+filename,noteindex])

    samplefreq = 44100
    N = 8000
    overlap = 0

    data = []
    labels = []

    for notefile, note in notefiles:
        rate, filedata = wav.read(notefile)
        if filedata.ndim == 1:
            for slice in read_wav_slices(N, overlap, filedata):
                data.append(slice)
                labels.append(note)
        if filedata.ndim == 2:
            for slice in read_wav_slices(N, overlap, filedata[:,0]):
                data.append(slice)
                labels.append(note)
            for slice in read_wav_slices(N, overlap, filedata[:,1]):
                data.append(slice)
                labels.append(note)
            
    data = np.vstack(data)
    np.save('data/processed/data.npy', data)
    np.save('data/processed/labels.npy', labels)