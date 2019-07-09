# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import PIL
from pydub import AudioSegment


# +
def file_to_mel_spectrum(file_path):  
    wav = AudioSegment.from_wav(file_path)
    wav = np.array(wav.get_array_of_samples()).astype(np.float32)
    # fmax is set low as higher frequencies are empty
    S = librosa.feature.melspectrogram(wav, sr=16000, n_mels=128, fmax=6000)
    return librosa.power_to_db(S, ref=np.max)

def show_spectrum(log_S, title):
    # Make a new figure
    plt.figure(figsize=(12,4))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=16000, x_axis='time', y_axis='mel')

    plt.title(title)

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    plt.tight_layout()
    
def save_spectrogram_image(log_S, image_path):
    # Make a new figure
    fig = plt.figure(figsize=(7, 7), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.tight_layout()

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=16000, ax=ax)
    fig.savefig(image_path)
    plt.close()


# -

feather_spectrum = file_to_mel_spectrum('feather_sound_recordings/feather_74_t_1.wav')
other_words_spectrum = file_to_mel_spectrum('feather_sound_recordings/noise_77_t_1.wav')
white_noise_spectrum = file_to_mel_spectrum('feather_sound_recordings/noise_28_t_1.wav')

# +
# import IPython
# IPython.display.Audio("feather_sound_recordings/feather_74_t_1.wav", autoplay=True)
# -

show_spectrum(feather_spectrum, 'Feather')
show_spectrum(other_words_spectrum, 'Smooth silky')
show_spectrum(white_noise_spectrum, 'White noise')


# Convert audio files to spectrograms

# +
def audio_files_to_spectrogram_image_files(source_dir, dest_dir):
    for audio_filename in os.listdir(source_dir):
        audio_path = os.path.join(source_dir, audio_filename)
        image_filename = '{}.png'.format(os.path.splitext(audio_filename)[0])
        image_path = os.path.join(dest_dir, image_filename)

        spectrum = file_to_mel_spectrum(audio_path)
        save_spectrogram_image(spectrum, image_path)

spectrogram_directory = 'feather_spectrograms'
if not os.path.exists(spectrogram_directory):
    os.makedirs(spectrogram_directory)
    
audio_files_to_spectrogram_image_files('feather_sound_recordings', spectrogram_directory)
# -

# What does one of the saved images look like?

from IPython.display import Image
Image(filename='feather_spectrograms/feather_64_5_t_1.png')

# ## Load spectrogram images

from fastai.vision import *
from fastai.metrics import error_rate

bs = 64

images_path = 'feather_spectrograms'
image_filenames = get_image_files(images_path)
image_filenames[:4]

np.random.seed(2)
pat = r'/([^_]+)_[^.]+.png$'

data = ImageDataBunch.from_name_re(
    images_path, image_filenames, pat, ds_tfms=None, size=224, bs=bs
)

data.show_batch(rows=20, figsize=(20, 20))

print(data.classes)

# ## Augment Resnet 34 classification model

# This doesn't work so hot...
#
# **Try learning more cycles/epochs!!!**

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)

learn.save('stage-1')

# +
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
# -

interp.plot_top_losses(9, figsize=(15,11), heatmap=False)

interp.plot_confusion_matrix(figsize=(5,5), dpi=60)

# ## Customise resnet weights

# This is a disaster, this model quickly finds a minimum in guessing that everything is noise 😞

learn.load('stage-1')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(4, max_lr=slice(1e-3,1e-1))

# +
interp_fine_tuned = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp_fine_tuned.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
# -

interp_fine_tuned.plot_confusion_matrix(figsize=(5,5), dpi=60)

# ## Normalised images

# Normalising the images and training for longer gives slightly better results, but the model is still having a terrible time 

normalised_data = ImageDataBunch.from_name_re(
    images_path, image_filenames, pat, ds_tfms=None, size=224, bs=bs
).normalize(imagenet_stats)

normalised_data.show_batch(rows=20, figsize=(20, 20))

learn = cnn_learner(normalised_data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(16)

# +
interp_normalised = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp_normalised.top_losses()

len(normalised_data.valid_ds)==len(losses)==len(idxs)
# -

interp_normalised.plot_confusion_matrix(figsize=(5,5), dpi=60)

interp_normalised.plot_top_losses(9, figsize=(15,11), heatmap=False)

losses, idxs = interp_normalised.top_losses()

losses, idxs

normalised_data.valid_ds.fnames

doc(ImageDataBunch)

normalised_data.valid_dl

normalised_data.path

dir(normalised_data.)











# ## Appendix - processing sound with scipy (getting bad spectra)

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('feather_sound_recordings/feather_74_t_1.wav')

plt.plot(samples)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


