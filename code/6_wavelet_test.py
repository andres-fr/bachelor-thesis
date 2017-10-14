"""This file implements the computation, saving, loading and plotting of the STFT and Wavelet
   representations of given audio files in .wav format. See the docstrings for more details.
"""

from __future__ import print_function
import numpy as np
import os



import scipy.io.wavfile as pywav
import pywt
import h5py



# problems with the backend among pyplot, librosa... pay atention to the order of this imports!
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
plt.ion() # turn matplotlib interactive mode on
# plt.rcParams["patch.force_edgecolor"] = True

import pylab


####################################################################################################
### FUNCTION DEFINITIONS
####################################################################################################

def load_all_wav_files(folder_path, expected_sample_rate, mono_only=True):
    """Given a path to a folder containing .wav files of expected_sample_rate, loads them into
       a dictionary, or throws an exception if any of them has different sample rate than expected.
       If mono_only is true (default), skip all non-mono files. Usage example:
       rock_pcm = load_all_wav_files("../datasets/rock/", 22050)
       merengue_pcm = load_all_wav_files("../datasets/merengue/", 22050)
    """
    d = {}
    wavpaths = [p for p in os.listdir(folder_path) if p.endswith(".wav")]
    for p in wavpaths:
        abs_p = os.path.abspath(folder_path+p)
        rate, arr = pywav.read(abs_p)
        if mono_only and (len(arr.shape) != 1): continue # if wanted mono_only, skip non-mono files
        if rate!=expected_sample_rate: # if mono file has unexpected sample rate, throw exception!
            raise RuntimeError("~load_all_wav_files: wrong samplerate in file "+abs_p+
                               ". Expected "+str(expected_sample_rate)+" but got "+str(rate))
        d[os.path.splitext(p)[0]] = arr # add the mono file with expected rate to the dictionary
    return d


####################################################################################################


def calculate_stft(data_array, fft_size=512, overlap_factor=0.5):
    """Given a 1D numpy array (usually the output of 'load_all_wav_files' with 'mono_only' enabled),
       retrieves a 2D numpy array containing the corresponding STFT. The 'fft_size' of each FFT
       window has to be a power of 2. The overlap_factor determines how much of each window is
       covered by the next window. Highly inspired from the code in:
       https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/
       Keep in mind that, if the PCM data has a depth of N bits, the corresponding STFT file will be
       64/N times bigger. Usage example:
       rock_stft = {k: calculate_stft(rock_pcm[k], fft_size=512) for k in rock_pcm}
       merengue_stft = {k : calculate_stft(merengue_pcm[k], fft_size=512) for k in merengue_pcm}
    """
    hop_size = np.int32(np.floor(fft_size * (1-overlap_factor)))
    pad_end_size = fft_size   # the last segment can overlap the end of the data array by no more than one window size
    total_segments = np.int32(np.ceil(len(data_array) / np.float32(hop_size)))
    window = np.hanning(fft_size)  # our half cosine window
    inner_pad = np.zeros(fft_size) # the zeros which will be used to double each segment size
    proc = np.concatenate((data_array, np.zeros(pad_end_size)))              # the data to process
    result = np.empty((fft_size, total_segments), dtype=np.float32)    # space to hold the result
    for i in xrange(total_segments):                      # for each segment
        current_hop = hop_size * i                        # figure out the current segment offset
        segment = proc[current_hop:current_hop+fft_size]  # get the current segment
        windowed = segment * window                       # multiply by the half cosine function
        padded = np.append(windowed, inner_pad)           # add 0s to double the length of the data
        spectrum = np.fft.fft(padded) / fft_size          # take the Fourier Transform and scale by the number of samples
        autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
        result[:, i] = autopower[:fft_size]               # append to the results array
    result = 20*np.log10(result)          # scale to db
    result = np.clip(result, -40, 200)    # clip values
    return result

def calculate_wavelet(data_array, kern="db1", wavelet_depth=None):
    """This is an alias for the wavedec() function of the pywt library, but further documented.
       Given a 1D numpy array (usually the output of 'load_all_wav_files' with 'mono_only' enabled),
       retrieves a list of 1D numpy arrays of different lengths, as usual in the Discrete Wavelet
       Transform. The 'kern' is the proper wavelet, can be chosen from pywt.wavelist() or crafted
       (see http://pywavelets.readthedocs.io/en/latest/ref/wavelets.html for more info). The depth
       of the transform can also be specified, the default behaviour is to let the dwt_max_level()
       function of the pywt library find an optimal depth automatically. Usage example:
       rock_wavelet = {k : calculate_wavelet(rock_pcm[k], kern="db1") for k in rock_pcm}
       merengue_wavelet = {k : calculate_wavelet(merengue_pcm[k], kern="db1") for k in merengue_pcm}
    """
    return pywt.wavedec(data_array, kern, level=wavelet_depth)


####################################################################################################


def export_stft_dataset(path, data):
    """Given a path as string, and the data as dict with numpy arrays, saves the dict to an .h5 file
       in the path. The arrays are stored as h5 datasets, to be retrieved with the corresponding
       key. The file can then be reloaded to pyton with the load_stft_dataset(path) function. If
       the input data is invalid will throw a RuntimeError. Usage example:
       export_stft_dataset("../datasets/rock_stft.h5", rock_stft)
       export_stft_dataset("../datasets/merengue_stft.h5", merengue_stft)
    """
    if type(data)!=dict:
        raise RuntimeError("~export_stft_dataset: data has to be a dict and was a "+str(type(data)))
    with h5py.File(path) as hf:
        for k,v in data.items():
            hf.create_dataset(k, data=data[k])

def export_wavelet_dataset(path, data):
    """Given a path as string, and the data as a dict of lists of numpy arrays (like the outputs of
       the calculate_wavelet() function), stores the data to an .h5 file in the path. The best way
       to load the dictionary back to python is to use the load_wavelet_dataset(path) function. If
       the input data is invalid will throw a RuntimeError. Usage example:
       export_wavelet_dataset("../datasets/rock_wavelet.h5", rock_wavelet)
       export_wavelet_dataset("../datasets/merengue_wavelet.h5", merengue_wavelet)
    """
    if (type(data)!=dict or any([type(x)!=list for x in data.values()])):
        raise RuntimeError("~export_wavelet_dataset: input data has to be a dict of str:list pairs")
    with h5py.File(path, "w") as hf:
        for k,v in data.items():
            g = hf.create_group(k)
            for idx, arr in enumerate(v):
                g[str(idx)] =arr


####################################################################################################


def load_stft_dataset(path):
    """Loads an .h5 file like the ones generated by the export_stft_dataset() function to a
       dictionary, where the keys are strings with the song names and the values 2D numpy arrays.
       It returns the dictionary. Usage example:
       rock_stft = load_stft_dataset("../datasets/rock_stft.h5")
       merengue_stft = load_stft_dataset("../datasets/merengue_stft.h5")
    """
    with h5py.File(path, "r") as hf:
        return {k:v.value for k,v in hf.items()}

def load_wavelet_dataset(path):
    """Loads an .h5 file like the ones generated by the export_wavelet_dataset() function to a
       dictionary, where the keys are strings with the song names and the values lists of numpy
       arrays. It returns the dictionary. Usage example:
       rock_wavelet = load_wavelet_dataset("../datasets/rock_wavelet.h5")
       merengue_wavelet = load_wavelet_dataset("../datasets/merengue_wavelet.h5")
    """
    with h5py.File(path, "r") as hf:
        return {song : [lst[str(i)].value for i in range(1+max([int(x) for x in lst]))]
                for song, lst in hf.items()}


####################################################################################################


def plot_stft(stft, title=None, color_map="gray"):
    """Given a 2D numpy array (as the ones generated by the calculate_stft() function), opens a new
       window and plots it as an image. The color map and the title can be customized.
       Usage example:
       for k,v in merengue_stft.items(): plot_stft(v, k)
       for k,v in rock_stft.items(): plot_stft(v, k)
    """
    fig = plt.figure()
    if title: fig.suptitle(title)
    plt.imshow(stft, cmap=color_map, origin="lower",
               aspect="auto", interpolation="nearest")

def plot_wavelet(tree, title=None, map_linear=True, color_map="gray", aspect_ratio=1.0/3):
    """Given a list of numpy arrays (as the ones generated by the calculate_wavelet() function),
       opens a new window and plots it as an image. The array contents are spreaded to fit the whole
       band, so the plotted regions of an array with 2 values will be twice as big as the regions of
       an array with 4 values, and so on. The color map and the title can be customized.
       Usage example:
       for k,v in merengue_wavelet.items(): plot_wavelet(v, k)
       for k,v in rock_wavelet.items(): plot_wavelet(v, k)
    """
    bottom = 0
    levels = len(tree)
    minval = abs(min([x.min() for x in tree])) + 1 # adding 1 to force logarithm to be >=0
    fig = plt.figure()
    if title: fig.suptitle(title)
    plt.gca().set_autoscale_on(False) #get current axis
    for idx, arr in enumerate(tree):
        scale = 1.0/levels if map_linear else 2.0**(idx-levels)
        arr += minval # avoid negative values
        arr = 20*np.log10(arr)
        #arr = np.clip(arr, -50, 300)    # handcrafted? to help better resolution
        plt.imshow([arr], cmap=color_map, aspect=aspect_ratio,
                   extent=[0,1,bottom, bottom+scale], interpolation="none")
        bottom += scale
    plt.show()





# ####################################################################################################
# ### EXAMPLES
# ####################################################################################################

# # load .wav files
# rock_pcm = load_all_wav_files("../datasets/rock/", 22050)
# merengue_pcm = load_all_wav_files("../datasets/merengue/", 22050)

# ####################################################################################################

# # calculate stfts from wav files
# rock_stft = {k: calculate_stft(rock_pcm[k], fft_size=512) for k in rock_pcm}
# merengue_stft = {k : calculate_stft(merengue_pcm[k], fft_size=512) for k in merengue_pcm}

# # calculate wavelets from wav files
# rock_wavelet = {k : calculate_wavelet(rock_pcm[k], kern="db1") for k in rock_pcm}
# merengue_wavelet = {k : calculate_wavelet(merengue_pcm[k], kern="db1") for k in merengue_pcm}

# ####################################################################################################

# # export the stfts to h5 files
# export_stft_dataset("../datasets/rock_stft.h5", rock_stft)
# export_stft_dataset("../datasets/merengue_stft.h5", merengue_stft)

# # export the wavelets to h5 files
# export_wavelet_dataset("../datasets/rock_wavelet.h5", rock_wavelet)
# export_wavelet_dataset("../datasets/merengue_wavelet.h5", merengue_wavelet)

# ####################################################################################################

# # reset the python session, load directly the h5 files, and check that everything is still OK
rock_stft = load_stft_dataset("../datasets/rock_stft.h5")
# merengue_stft = load_stft_dataset("../datasets/merengue_stft.h5")
rock_wavelet = load_wavelet_dataset("../datasets/rock_wavelet.h5")
# merengue_wavelet = load_wavelet_dataset("../datasets/merengue_wavelet.h5")

# print(rock_stft["trouble"].shape)
# print(merengue_stft["movidito"].shape)
# print(rock_wavelet["trouble"][0].dtype)
# print()merengue_wavelet["movidito"][0])


# ####################################################################################################

# # plot the stft of all 20 merengue songs
# for k,v in merengue_stft.items(): plot_stft(v, k)

# # plot the stft of all 20 rock songs
for k,v in rock_stft.items()[11:17]: plot_stft(v, k)

# plot the wavelet of all 20 merengue songs
# for k,v in merengue_wavelet.items(): plot_wavelet(v, k)

# # plot the wavelet of all 20 rock songs
for k,v in rock_wavelet.items()[11:17]: plot_wavelet(v, k)

raw_input("holding process...")
