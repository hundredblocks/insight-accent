import wave

import librosa
import numpy as np

N_FFT = 2048


def slice(infile, outfilename, start_ms, end_ms):
    width = infile.getsampwidth()
    rate = infile.getframerate()
    frames_per_ms = rate / 1000
    length = (end_ms - start_ms) * frames_per_ms
    start_index = start_ms * frames_per_ms

    out = wave.open(outfilename, "w")
    out.setparams((infile.getnchannels(), width, rate, length, infile.getcomptype(), infile.getcompname()))

    infile.rewind()
    anchor = infile.tell()
    infile.setpos(anchor + start_index)
    out.writeframes(infile.readframes(length))


# Reads wav file and produces spectrum
# Fourier phases are ignored
def read_audio_spectrum(x, fs, n_fft=N_FFT, reduce_factor=1):
    x = x[0:len(x) / reduce_factor]
    S = librosa.stft(x, n_fft, hop_length=n_fft/4)
    # p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S, fs
