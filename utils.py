import wave

import librosa
import numpy as np
import os
import python_speech_features

from matplotlib import pyplot as plt

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


def multislice(infile, outfilepath, outfilename, ms_cut_size=3000, ms_step_size=1, start_pad=0, end_pad=None):
    # width = infile.getsampwidth()
    rate = infile.getframerate()
    nframes = infile.getnframes()
    total_length_seconds = nframes / float(rate)
    total_length_ms = total_length_seconds * 1000

    if end_pad is None:
        end_pad = 0
    to_cut_length = (total_length_ms - end_pad) - start_pad

    if to_cut_length < ms_cut_size:
        return

    num_cuts = int(1 + (to_cut_length - ms_cut_size) // ms_step_size)

    for i in range(num_cuts):
        start_ms = i * ms_step_size + start_pad
        end_ms = start_ms + ms_cut_size
        name_and_extension = outfilename.split('.')
        name_str = "_".join(name_and_extension[:-1])
        cut_outname = '%s_%s-%s.%s' % (name_str, start_ms, end_ms, name_and_extension[-1])
        slice(infile, os.path.join(outfilepath, cut_outname), start_ms=start_ms, end_ms=end_ms)


# Reads wav file and produces spectrum
# Fourier phases are ignored
def read_audio_spectrum(x, fs, n_fft=N_FFT, reduce_factor=1):
    x = x[0:len(x) / reduce_factor]
    S = librosa.stft(x, n_fft, hop_length=n_fft / 4)
    # p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S


def plot_spectrum(x, fs, cut=620):
    spec = read_audio_spectrum(x, fs)
    spec_cut = spec[:cut, :]
    spec_inv = np.array([spec_cut[-(i + 1)] for i, a in enumerate(spec_cut)])
    plt.figure()
    plt.imshow(spec_inv)
    plt.show()


def fft_to_audio(out_name, spectrogram, sampling_frequency, n_fft=N_FFT, n_iter=500, entire_path=False):
    p = 2 * np.pi * np.random.random_sample(spectrogram.shape) - np.pi
    for i in range(n_iter):
        S = spectrogram * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, n_fft))

    if not entire_path:
        output_filename = 'outputs/' + out_name
    else:
        output_filename = out_name
    librosa.output.write_wav(output_filename, x, sampling_frequency)
    print output_filename
    return output_filename


def get_mfcc(x, fs, nfft):
    mfcc = python_speech_features.mfcc(x, samplerate=fs, numcep=24, nfft=nfft)
    return mfcc


def plot_all(a_content, a_style, final_result, initial_spectrogram):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.title('Content')
    plt.imshow(a_content[:400, :])
    plt.subplot(1, 4, 2)
    plt.title('Style')
    plt.imshow(a_style[:400, :])
    plt.subplot(1, 4, 3)
    plt.title('Result')
    plt.imshow(final_result[:400, :])
    plt.subplot(1, 4, 4)
    plt.title('Initial Noise Vector')
    plt.imshow(initial_spectrogram[:400, :])
    plt.show()
