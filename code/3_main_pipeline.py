"""
"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import random
import time
import datetime
from itertools import cycle
import pickle
import h5py
import os
import sys
import scipy.io.wavfile as pywav


# other
from PIL import Image
from six.moves import xrange
from tabulate import tabulate
import io




# support parallelization
import concurrent
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor #, ProcessPoolExecutor

# problems with the backend among pyplot, librosa... pay atention to the order of this imports!
import matplotlib
matplotlib.use('agg')
import librosa
import librosa.display
import matplotlib.pyplot as plt
plt.ioff() # turn matplotlib interactive mode off
plt.rcParams["patch.force_edgecolor"] = True






####################################################################################################
### GLOBAL PREPROC. SETTINGS
####################################################################################################

#
DATASETS_PATH = "../datasets/"
LOG_PATH = "../tensorboard_logs/"
#
CARNATIC_MP3_PATH = DATASETS_PATH+"carnatic_mp3/" # an existing folder with all 2455 mp3 recordings
CARNATIC_WAV_PATH = DATASETS_PATH+"carnatic_wav_22050/" # an existing but empty folder
SAMPLERATE = 22050  # the desired samplerate for mp3->wav conversion. if 8000, result has 15GB
#
CLEAN_LABELS_PICKLE = DATASETS_PATH+"labels_dict.pickle" # this labels were cleaned in 2_preprocess_carnatic.py
WAV_22050_H5PATH = DATASETS_PATH+"wavs_22050.h5"
#
CQT_H5PATH = DATASETS_PATH+"cqts_22050_4096_4.h5"
MEL_H5PATH = DATASETS_PATH+"melgrams_22050_4096_4.h5"
FFT_SIZE = 512*4
HOP = int(FFT_SIZE/4)
N_MELS = 128


####################################################################################################
### GENERAL STUFF
####################################################################################################


def prompted_warning(message):
    print("\nWARNING:", message)
    answer = str(raw_input("continue? (press 'Y' or 'y' for yes, anything else for no)\n")) # str for the paranoics
    if answer in "Yy":
        return
    else:
        print("terminating...")
        quit()


def int_to_humantime(secs, downsample_ratio=1):
    secs /= downsample_ratio
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return "%dh%02dm%02ds"%(h, m, s)

def load_pickle(inpath):
    with open(inpath, "rb") as f:
        return pickle.load(f)


def slice_lastdimension(arr, from_idx_included, till_idx_excluded):
    arr = np.moveaxis(arr, -1, 0) # move last dim to the fromt
    arr = arr[from_idx_included:till_idx_excluded] # slice the axis
    arr = np.moveaxis(arr,0, -1) # bring sliced axis back to its original pos.
    return arr


####################################################################################################
### CONVERT THE MP3 FILES TO MONO WAV
####################################################################################################

def convert_mp3_to_wav(mp3_filepath, output_folder, samplerate):
    """Using ffmpeg, which has to be installed, this function takes the absolute path to an mp3
       file and copies it to the given output folder as a mono WAV file with the specified sample
       rate. This function is parallelizable (see convert_all_mp3_to_wav, also for usage example).
    """
    # parse filenames and make shell command
    mp3_name = os.path.splitext(os.path.basename(mp3_filepath))[0]
    wav_filepath = output_folder+mp3_name+".wav"
    convert_command = ("ffmpeg -i "+mp3_filepath+" -ac 1 -ar "+str(samplerate)+" "+ wav_filepath +
                       " < /dev/null > /dev/null 2>&1")
    # issue the command and report result
    if(os.system(convert_command) != 0):
        raise RuntimeError("~convert_mp3_to_wav: ERROR calling "+ convert_command)
    else:
        print("converted file", mp3_filepath, "to file", wav_filepath)


def convert_all_mp3_to_wav(from_path, to_path, samplerate, num_threads=20):
    """This function copies all the files in the from_path folder (expected to hold mp3 files only)
       to the to_path folder as WAV mono format with the given samplerate. To do so, it calls
       convert_mp3_to_wav for every file in parallel (multithreaded). Usage example:
       convert_all_mp3_to_wav(CARNATIC_MP3_PATH, CARNATIC_WAV_PATH, SAMPLERATE)
    """
    # list with the absolute paths to mp3 files
    filenames = [os.path.join(from_path, n) for n in os.listdir(from_path)]
    # create output directory if doesn't exist:
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    # call convert_mp3_to_wav for every file in filenames, parallelized
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(lambda x: convert_mp3_to_wav(x, to_path, samplerate), filenames)

### DO THIS ONLY ONCE PER SAMPLERATE
# convert_all_mp3_to_wav(CARNATIC_MP3_PATH, CARNATIC_WAV_PATH, SAMPLERATE)


####################################################################################################
### SAVE WAVS AS H5 DICT OF DICTS
####################################################################################################

def load_wav_file(absolutepath, expected_samplerate=SAMPLERATE, mono_only=True):
    """Given a path to a .wav file of expected_sample_rate, and a boolean asking if the file is
       expected to be mono, returns the file contents as 1D numpy float array, or throws an
       exception if samplerate or number of channels are different than expected, or if file not
       found. Usage example:
           song = load_wav_file(CARNATIC_WAV_PATH+"song.wav", 8000, True)
    """
    rate, arr = pywav.read(absolutepath)
    if rate!=expected_samplerate:
        raise RuntimeError("~load_wav_file: expected samplerate was "+str(expected_samplerate)+
                           " but got "+str(rate))
    if mono_only and (len(arr.shape) != 1):
        raise RuntimeError("~load_wav_file: file expected to be mono but wasn't! "+absolutepath)
    return arr


def save_all_raaga_wavs_in_folder_to_h5_file(labels, from_path, to_path):
    """Given the dataset labels as a dictionary of key=rec_id, value=label_dict, the path to the
       wavs folder, and the output path for the h5py file, this function loads all wav files that
       have a raaga specified in their labels, and stores them as an int16 numpy array in the h5
       file, under the following path: 'raaga_uuid/rec_mbid'. This way, they can be found by raaga.
       Usage example:
       save_all_raaga_wavs_in_folder_to_h5_file(load_pickle(CLEAN_LABELS_PICKLE), CARNATIC_WAV_PATH)
    """
    with h5py.File(to_path, "w") as hf:
        raaga_labels = {k:v for k,v in labels.iteritems() if v["raaga"]}
        size = len(raaga_labels)
        for i, mbid in enumerate(raaga_labels):
            lbl = raaga_labels[mbid]
            raaga = lbl["raaga"][0]["uuid"]
            wavpath = os.path.join(from_path, mbid+".wav")
            arr = load_wav_file(wavpath) # type: int16
            h5path = raaga+"/"+mbid
            print(i+1,"from", size, ">>> now dumping:", h5path)
            hf.create_dataset(h5path, data=arr, compression="lzf")

def traverse_ddict(ddict, fn=lambda cl,elt,data: print(cl, elt, data.shape)):
    for cl, elts in ddict.iteritems():
            for elt, data in elts.iteritems():
                fn(cl, elt, data)

def traverse_h5_ddict(h5path, fn=lambda cl,elt,data: print(cl, elt, data.shape)):
    """This function is just to exemplify how to traverse a h5 file that stores the recordings by
       raaga: it prints the path and the shape of the content. Usage example:
         traverse_h5_raaga_file(WAV_22050_H5PATH)
    """
    with h5py.File(h5path, "r") as hf:
        traverse_ddict(hf, fn)


# ### DO THIS ONLY ONCE PER SAMPLERATE
# save_all_raaga_wavs_in_folder_to_h5_file(load_pickle(CLEAN_LABELS_PICKLE), CARNATIC_WAV_PATH,
#                                          WAV_22050_H5PATH)

####################################################################################################
### FROM THE WAV H5 DATASET, EXTRACT THE CQT AND MFCC REPRESENTATIONS AND SAVE THEM TO H5 DATASET
####################################################################################################


def make_cqt_from_arr(arr, samplerate, winsize, hopsize,
                      bins_per_octave=36, fmin=librosa.note_to_hz("G2"), octaves=5):
    """
    """
    cqt = librosa.cqt(arr.astype(np.float32), sr=samplerate, fmin=fmin,hop_length=hopsize,
                           n_bins=int(octaves*bins_per_octave), bins_per_octave=bins_per_octave)
    return np.abs(cqt).astype(np.float32)

def make_melgram_from_arr(arr, winsize, hopsize, fmin=librosa.note_to_hz("G2"), n_mels=128):
    """
    """
    stft = np.abs(librosa.core.stft(arr.astype(np.float32), n_fft=winsize, hop_length=hopsize), dtype=np.float32)
    mel = librosa.feature.melspectrogram(S=stft, n_mels=n_mels, fmin=fmin)
    return mel.astype(np.float32)

def create_cqt_h5dict_from_wav_h5dict(wav_h5path, cqt_h5path, samplerate, winsize, hopsize,
                                      num_threads=12):
    """
    """
    with h5py.File(wav_h5path, "r") as hfwav, h5py.File(cqt_h5path, "w") as hfcqt:
        i = 1
        size = sum([len(v.keys()) for v in hfwav.values()])
        ragrecs = [(raaga, rec) for raaga in hfwav for rec in hfwav[raaga]]
        makecqt = lambda ragrec: (ragrec, make_cqt_from_arr(hfwav[ragrec[0]][ragrec[1]].value,
                                                              samplerate, winsize, hopsize))
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for ragrec, cqt in executor.map(makecqt, ragrecs):
                print(i,"from", size, ">>> calculating and exporting CQT to", cqt_h5path)
                h5path = ragrec[0]+"/"+ragrec[1]
                hfcqt.create_dataset(h5path, data=cqt, compression="lzf")
                i += 1

def create_mel_h5dict_from_wav_h5dict(wav_h5path, mel_h5path, winsize, hopsize, n_mels,
                                      num_threads=12):
    """
    """
    with h5py.File(wav_h5path, "r") as hfwav, h5py.File(mel_h5path, "w") as hfmel:
        i = 1
        size = sum([len(v.keys()) for v in hfwav.values()])
        ragrecs = [(raaga, rec) for raaga in hfwav for rec in hfwav[raaga]]
        makemel = lambda ragrec: (ragrec, make_melgram_from_arr(hfwav[ragrec[0]][ragrec[1]].value,
                                                                  winsize, hopsize, n_mels=n_mels))
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for ragrec, mel in executor.map(makemel, ragrecs):
                print(i,"from", size, ">>> calculating and exporting melgram to", mel_h5path)
                h5path = ragrec[0]+"/"+ragrec[1]
                hfmel.create_dataset(h5path, data=mel, compression="lzf")
                i += 1

# create_mel_h5dict_from_wav_h5dict(WAV_22050_H5PATH, MEL_H5PATH, FFT_SIZE, HOP, N_MELS)
# create_cqt_h5dict_from_wav_h5dict(WAV_22050_H5PATH, CQT_H5PATH, SAMPLERATE, FFT_SIZE, HOP)




####################################################################################################
### GET GULATI'S TEST SET (SEE ALSO HISTOGRAM)
####################################################################################################

# MY DATASET: the raaga_labels have a total of 2050 recordings, a duration of 1531164.709 seconds (425.32353 hours)


def load_gulati_test_labels():
    """reads one txt file with ragaid raganame lines, and other with recpath recid ragaid lines,
       and returns a dictionary of the form recid : (ragname, [ragids...], path).
       ragids is a list because the raaga Purvikayani has several *unique* IDs...
    """
    import codecs
    with codecs.open("ragaid_to_raganame.txt", "r", "utf-8") as ragmap,\
         codecs.open("path_mbid_ragaid.txt", "r", "utf-8") as labels:
        raglist = [x.rstrip("\n").split("\t") for x in ragmap if x[0] not in "# \n"]
        rag2name = {z[0]:z[1] for z in raglist}
        name2rags = {z[1]:[] for z in raglist} # ragname:ragid
        for ragid, ragname in raglist:
            name2rags[ragname].append(ragid)
        lbl_list = [y.rstrip("\n").split("\t") for y in labels if y[0] not in "# \n"] # path, recid raagid
        return {recid:(rag2name[ragid], name2rags[rag2name[ragid]], path) for path, recid, ragid in lbl_list}


def load_dunya_raaga_lists():
    """returns a list of (raaga_id, raaga_name) tuples, for the raagas present in the
       dunya dataset, ordered by decreasing total size,
       and a list with the indexes of the classes present in Gulati's test dataset
    """
    import codecs
    # open the raaga map file
    with codecs.open("./all_raagas_ordered_by_size.txt", "r", "utf-8") as f:
        raagas_ordered_by_size = [tuple(line.rstrip('\n').split("\t")) for line in f
                                  if line[0] not in "# \n"]# list of [ragid, ragname] lists
    gulati_testlabels = load_gulati_test_labels()
    gulati_ragnames = {x[0] for x in gulati_testlabels.values()}
    raagas_gulati_ordered_by_size = [r for r in raagas_ordered_by_size if r[1] in gulati_ragnames]
    return raagas_ordered_by_size, raagas_gulati_ordered_by_size

def load_gulati_test_ddict(h5path):
    """The labels in Gulati's test set are mapped to one single raaga name, but to multiple raaga ids.
       On the other side, the labels in the dunya h5 dataset are mapped by unique raaga ids.
       So this function returns gulati's test set labels as a dict of key=raaga_id, value=set_of_rec_ids,
       with 2 things to keep in mind:
          1) some recordings may not be found
          2) when multiple raaga_ids existing, the one of the h5 dataset is chosen
       It also returns a dict of key=raaga_id, val=total_class_width and a list with the Gulati's
       rec_ids that couldn't be found in the dunya h5 file.
    """
    with h5py.File(h5path, "r") as hf:
        # map raaga_name to dunya's raaga_id, and dunya's rec_id to dunya's raaga
        ragname2ragid = {ragname:ragid for ragid, ragname in load_dunya_raaga_lists()[0]}
        dunya_rec2rag = {recid:raag for raag, recs in hf.iteritems() for recid in recs.keys()}
        # get all gulati test set labels
        gulati_testlabels = load_gulati_test_labels() # recid : (ragname, [ragids...], path)
        # dict of key=raaga_id, value=set_of_rec_ids. The raaga is dunyas, the recs are from gulati
        result = {ragname2ragid[x[0]]:set() for x in gulati_testlabels.values()}
        not_found = [] # list with recs that weren't found in dunya
        for recid, (ragname, ragids, _) in gulati_testlabels.iteritems():
            if ragname2ragid[ragname] not in ragids: # this condition should never happen
                prompted_warning("in load_gulati_test_dict: inconsistent label for rec "+recid)
            try: # if the recid is found in the dunya_rec2rag dict, means that it exists in the h5 file
                result[dunya_rec2rag[recid]].add(recid)
            except KeyError: # in this case Gulati's label is not present in the dunya h5 DS
                print(len(not_found)+1, "not found: ragname, recid =", ragname, recid)
                not_found.append(recid)
        # finally print testset stats and a warning, and return
        print("\nLOADING GULATI'S TEST LABELS FROM"+str(h5path)+":")
        raglengths = {k:sum([hf[k][vv].shape[-1] for vv in v]) for k,v in result.iteritems()}
        norm = sum(raglengths.values())/float(100)
        for k,v in result.iteritems():
            print("raaga_id:", k, "number of elements:", len(v), "\trel. duration:",
                  "{:10.2f}".format(raglengths[k]/norm)+"%")
        if not_found:
            prompted_warning("in load_gulati_test_ddict: "+str(len(not_found))+
                             " recordings weren't found in the database.")
        return result, raglengths, not_found


####################################################################################################
### DEFINE CARNATIC LOADING: LOAD GULATI'S TEST SET, AND THE REST AS TRAIN SET
####################################################################################################


def load_labels(h5path, raaga_list):
    with h5py.File(h5path, "r") as hf:
        return [(raag, rec, hf[raag][rec].shape) for raag in raaga_list for rec in hf[raag]]

def label_stats(label_list, prettyprint_title=""):
    label_list = sorted(label_list)
    raagas = {x[0]:[0,0] for x in label_list}
    for raag, rec, shape in label_list:
        n, raagdur = raagas[raag]
        raagas[raag][0] += 1
        raagas[raag][1] += shape[-1]
    if prettyprint_title:
        print(prettyprint_title)
        for k,v in raagas.iteritems():
            print("  raaga, n.recs, total frames:", k, v[0], v[1])
    return raagas

def split_labels(label_list, ratio_first_to_total, print_stats=True):
    """Given a list of (raag,rec, shape) labels, splits it into 2 lists, by following a Bernoulli distribution:
       If the random number is less than ratio the label goes to the first list, else to the second.
       Note that this way of splitting a dataset into two disjoint datasets is based on the
       number of different recordings, and NOT on their duration.
    """
    if(ratio_first_to_total<0 or ratio_first_to_total>1):
        raise RuntimeError("~split_labels: 0<=ratio<=1 required, but was " +str(ratio_first_to_total))
    labels_copy = label_list[:] # shallow copy
    labels1 = []
    labels2 = []
    while labels_copy:
        if (np.random.random()<ratio_first_to_total):
            labels1.append(labels_copy.pop())
        else:
            labels2.append(labels_copy.pop())
    raagas_l1 = set([x[0] for x in labels1])
    raagas_l2 = set([x[0] for x in labels2])
    if (len(raagas_l1)!=len(raagas_l2)):
        prompted_warning("in split_labels: some raagas have no representation")
    if print_stats:
        label_stats(labels1, "LABELS 1:")
        label_stats(labels2, "LABELS 2:")
    return labels1, labels2

def load_ddict(h5path, label_list):
    """label_list is a list of (raag, rec, shape) tuples
    """
    result = {x[0]:{} for x in label_list}
    with h5py.File(h5path, "r") as hf:
        for raag, rec, _ in label_list:
            result[raag][rec] = hf[raag][rec].value
        return result

def load_carnatic_as_ddicts(h5path, raaga_list):
    """Given a path to a h5 ddict dataset and a list of raaga_ids, returns two disjoint ddicts,
       which hold the raaga_id->rec_id->rec *for the classes given in the raaga_list*:
       (train_ddict, test_ddict), where test_ddict holds all the recs of Gulati's test set that
       were found in the h5 dataset, and the train_ddict the remaining ones.
    """
    # load labels and split them into train, cv, test
    labels = load_labels(h5path, raaga_list) # list of (raag, rec, hf[raag][rec].shape)
    label_stats(labels, "TOTAL LABELS")
    #
    gulati_test_ddict, _, _ = load_gulati_test_ddict(h5path)
    gulati_recs = {rec for recs in gulati_test_ddict.values() for rec in recs}
    test_labels = [l for l in labels if l[1] in gulati_recs]
    train_labels = [l for l in labels if l[1] not in gulati_recs]
    # load the three separate datasets as dicts of dicts
    train_ddict = load_ddict(h5path, train_labels)
    test_ddict = load_ddict(h5path, test_labels)
    # test that they are disjoint:
    traverse_ddict(train_ddict, lambda cl,elt,data:
                   prompted_warning("found sample in both training and test set! "+str(cl)+" "+str(elt))\
                   if test_ddict.get(cl, {}).get(elt, False) else None)
    # and return them
    return train_ddict, test_ddict



def load_gulati_wavdata(wav_h5path, class_set, train_ratio=0.5):
    """given a path to an h5 file holding raaga_id->rec_id->1Darray, where the 1d array is the time-
       domain audio signal, loads and returns its subset corresponding to Gulati's test labels
       as a dict of dicts.
    """
    with h5py.File(wav_h5path, "r") as hf:
        gul_labels, _, _ = load_gulati_test_ddict(wav_h5path)
        gul_labels = {k:v for k,v in gul_labels.iteritems() if k in class_set}
        gul_label_list = [(rag, rec, hf[rag][rec].shape) for rag, recs in gul_labels.iteritems() for rec in recs]
        train_dict = {r:{} for r in gul_labels}
        test_dict = {r:{} for r in gul_labels}
        train_labels, test_labels =  split_labels(gul_label_list, train_ratio, print_stats=True)
        for i, (rag, rec, shape) in enumerate(train_labels):
            train_dict[rag][rec] = hf[rag][rec].value
            print(i, "\tadded to training set", rag, rec, int_to_humantime(shape[-1], SAMPLERATE))
        for i, (rag, rec, shape) in enumerate(test_labels):
            test_dict[rag][rec] = hf[rag][rec].value
            print(i, "\tadded to test set", rag, rec, int_to_humantime(shape[-1], SAMPLERATE))
        return train_dict, test_dict

####################################################################################################
### SAVE DATASETS TO PNG IMAGES
####################################################################################################


def export_ddict_to_pngs(ddict, outfolder, chunksize=None):
    """
    """
    for class_id, samples in ddict.iteritems():
        classfolder = os.path.join(outfolder, class_id)
        if not os.path.exists(classfolder):
            os.makedirs(classfolder)
        for sample_id, data in samples.iteritems():
            rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
            width = data.shape[1]
            chunksize = chunksize if chunksize else width
            for i in xrange(0, width+1-chunksize, chunksize):
                chunkpath = os.path.join(classfolder, str(sample_id)+"_"+str(i)+"_"+str(i+chunksize)+".png")
                print("export_ddict_to_pngs: saving image to", chunkpath)
                im = Image.fromarray(rescaled[:, i:i+chunksize])
                im.save(chunkpath)


def load_ddict_and_export_to_pngs(dd_h5path, outpath, chunksize=None):
    """USAGE EXAMPLE:
       MEL_H5PATH = "../datasets/melgrams_22050_4096_4.h5"
       CQT_H5PATH = "../datasets/cqts_22050_4096_4.h5"
       load_ddict_and_export_to_pngs(MEL_H5PATH, os.path.join(DATASETS_PATH, "melgrams_png"), chunksize=128)
       load_ddict_and_export_to_pngs(CQT_H5PATH, os.path.join(DATASETS_PATH, "cqts_png"), chunksize=180)
    """
    _, gulati_raagas = load_dunya_raaga_lists()
    class_set = [x[0] for x in gulati_raagas]
    tr, tst = load_carnatic_as_ddicts(dd_h5path, class_set)
    export_ddict_to_pngs(tr, os.path.join(outpath, "train"), chunksize)
    export_ddict_to_pngs(tst, os.path.join(outpath, "test"), chunksize)

# MEL_H5PATH = "../datasets/melgrams_22050_4096_4.h5"
# CQT_H5PATH = "../datasets/cqts_22050_4096_4.h5"
# load_ddict_and_export_to_pngs(MEL_H5PATH, os.path.join(DATASETS_PATH, "melgrams_png"), chunksize=128)
# load_ddict_and_export_to_pngs(CQT_H5PATH, os.path.join(DATASETS_PATH, "cqts_png"), chunksize=180)



####################################################################################################
### DEFINE TF MODELS *****
####################################################################################################


# ### ALIASES

# # W B INITIALIZATION: HE(2015)-> DELVING DEEP INTO RECTIFIERS... PAGE 4, POINT 10
# # weights with small noise for symmetry breaking and close to zero to avoid zero-gradients
# def weight_stddev(shape):
#     """calculates sqrt(2.0/prod(shape)). See
#        HE(2015)-> DELVING DEEP INTO RECTIFIERS... PAGE 4, POINT 10
#     """
#     return np.sqrt(2.0/np.prod(shape))


# fully_weight_var = lambda shape: tf.Variable(tf.truncated_normal(shape,stddev=weight_stddev(shape)))
# conv_weight_var = lambda shape: tf.Variable(tf.truncated_normal(shape,
#                                                                 stddev=weight_stddev(shape[0:3])))
# # bias greater than zero to avoid dead neurons, since using ReLU
# bias_variable = lambda shape: tf.Variable(tf.constant(0.0, shape=[shape]))

# aliases
relu = tf.nn.relu
conv1d = tf.nn.conv1d
conv2d = tf.nn.conv2d
max_pool = tf.nn.max_pool
matmul = tf.matmul
dropout = tf.nn.dropout
l2loss = tf.nn.l2_loss
batch_norm = tf.layers.batch_normalization



# weights with small noise for symmetry breaking and close to zero to avoid zero-gradients
weight_variable = lambda shape, stddev=0.1: tf.Variable(tf.truncated_normal(shape, stddev=stddev))
# bias greater than zero to avoid dead neurons, since using ReLU
bias_variable = lambda shape: tf.Variable(tf.constant(0.1, shape=[shape]))
# conv has stride of one and are zero padded so that the output is the same size as the input
conv2dlayer = lambda x, W: conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# plain old max pooling over 2x2 blocks
max_pool_2x2 = lambda x: max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
max_pool_2x3 = lambda x: max_pool(x, ksize=[1, 2, 3, 1], strides=[1, 2, 3, 1], padding='SAME')





def conv2d_layer(batch, conv_height, conv_width, conv_outchans,
                 maxpool_height, maxpool_width,
                 with_relu=False,
                 droprate_placeholder=None,
                 with_batchnorm=False,
                 is_training_placeholder=None,
                  verbose=True, padding="SAME"):
    """returns a tuple with (PIPELINE,  l2loss) where pipeline is conv->batchnorm->relu->pool
       if with_batchnorm is true or conv->relu->pool else, and the second is the l2 loss of the
       kernel weights used in the conv2d layer.
    """
    batch_n, batch_h, batch_w, batch_ch = batch.get_shape().as_list()
    # W = conv_weight_var([conv_height, conv_width, batch_ch, conv_outchans])
    W = weight_variable([conv_height, conv_width, batch_ch, conv_outchans])
    b = bias_variable(conv_outchans)
    conv = conv2d(batch, W, strides=[1,1,1,1], padding=padding) +b
    if with_batchnorm:
        if is_training_placeholder is None:
            prompted_warning("in conv2d_layer: with_batchnorm was True but no is_training_placeholder was given.")
        conv = batch_norm(conv, center=True, scale=True, training=is_training_placeholder)
    if with_relu:
        conv=relu(conv)
    pool = max_pool(conv, ksize=[1,maxpool_height,maxpool_width,1],
                    strides=[1,maxpool_height,maxpool_width,1], padding=padding)
    if droprate_placeholder is not None:
        pool = dropout(pool, droprate_placeholder)
    if verbose:
        print("conv2d_layer output shape:", pool.get_shape().as_list())
    return pool, l2loss(W)

def flatten_batch(batch):
    batch_shape = batch.get_shape().as_list()
    return tf.reshape(batch, [-1, np.prod(batch_shape[1:])])

def fully_connected_layer(batch, out_size, with_relu=False, droprate_placeholder=None, verbose=True):
    batch_n, in_size = batch.get_shape().as_list()
    W = weight_variable([in_size, out_size])
    b = bias_variable(out_size)
    fc = matmul(batch, W) + b
    if with_relu:
        fc = relu(fc)
    if droprate_placeholder is not None:
        fc = dropout(fc, droprate_placeholder)
    if verbose:
        print("fully_connected_layer output shape:", fc.get_shape().as_list())
    return fc, l2loss(W)

################ MNIST MODELS

def wav_simple_model(batch, num_classes, droprate_placeholder,
                         is_training_placeholder,
                         with_batchnorm=False, basenum_kernels=8, hidden_size=64):
    """A conv network for batch inputs of shape [None, 22050]
    """
    # preprocess: convert the input img tensor to 4D (required by the conv2d function)
    _, batch_l = batch.get_shape().as_list()
    batch_expanded = tf.reshape(batch, [tf.shape(batch)[0], 1, batch_l, 1])
    print("batch expanded shape:", batch_expanded.get_shape().as_list())
    # representation learning
    layer1, reg1 = conv2d_layer(batch_expanded, 1,5,2, 1,5, with_relu=True, droprate_placeholder=droprate_placeholder)
    layer2, reg2 = conv2d_layer(layer1, 1,5,4, 1,5, with_relu=True, droprate_placeholder=droprate_placeholder)
    layer3, reg3 = conv2d_layer(layer2, 1,5,8, 1,5, with_relu=True, droprate_placeholder=droprate_placeholder)
    layer4, reg4 = conv2d_layer(layer3, 1,5,16, 1,4, with_relu=True, droprate_placeholder=droprate_placeholder)
    layer5, reg5 = conv2d_layer(layer4, 1,5,32, 1,4, with_relu=True, droprate_placeholder=droprate_placeholder)
    layer6, reg6 = conv2d_layer(layer5, 1,5,64, 1,3, with_relu=True, droprate_placeholder=droprate_placeholder)
    layer7, reg7 = conv2d_layer(layer6, 1,5,128, 1,2, with_relu=True, droprate_placeholder=droprate_placeholder)
    layer8, reg8 = conv2d_layer(layer7, 1,5,256, 1,2, with_relu=True, droprate_placeholder=droprate_placeholder)
    # fully connected top classifier
    conv_out_flat = flatten_batch(layer8)
    fcX, regX = fully_connected_layer(conv_out_flat, hidden_size,
                                      with_relu=True, droprate_placeholder=droprate_placeholder)
    logits, regY = fully_connected_layer(fcX, num_classes, with_relu=False)
    l2reg = reg1+reg2+reg3+reg4+reg5 + regX+regY
    return logits, l2reg

def mnist_model_conv(batch, num_classes, droprate_placeholder, is_training, hidden_size=1024):
    """A simple fully conv network:
       input:       28x28
       reshaped to 28x28x1
       =========LAYER1========
       conv   5x5   28x28x32
       maxpool2x2   14x14x32
       =========LAYER2========
       conv   5x5   14x14x64
       maxpool2x2   7x7x64
       =========FCN1========
       fully        (7*7*64) x hidden_size
       =========TOP===========
       fully        hidden_size x num_classes
    """
    # preprocess: convert the input img tensor to 4D (required by the conv2d function)
    batch_expanded = tf.expand_dims(batch, axis=3)
    print("batch expanded shape:", batch_expanded.get_shape().as_list())
    # representation learning
    layer1, reg1 = conv2d_layer(batch_expanded, 5,5,32, 2,2, with_relu=True)
    layer2, reg2 = conv2d_layer(layer1, 5,5,64, 2,2, with_relu=True)
    # fully connected top classifier
    conv_out_flat = flatten_batch(layer2)
    fc1, reg3 = fully_connected_layer(conv_out_flat, hidden_size,
                                      with_relu=True, droprate_placeholder=droprate_placeholder)
    logits, reg4 = fully_connected_layer(fc1, num_classes, with_relu=False)
    l2reg = reg1+reg2+reg3+reg4
    return logits, l2reg


################# CARNATIC MODELS
def carnatic_model_basic(batch, num_classes, droprate_placeholder,
                         is_training_placeholder,
                         with_batchnorm=False, basenum_kernels=8, hidden_size=1024):
    """A simple fully conv network:
       input:       128X860
       reshaped to  128x860x1
       =========LAYER1========
       conv   5x5   128x860xbasenum_kernels
       maxpool2x2    64x430xbasenum_kernels
       =========LAYER2========
       conv   5x5    64x430xbasenum_kernels*2
       maxpool2x2    32X215Xbasenum_kernels*2
       =========LAYER3========
       conv   5x5    32x215xbasenum_kernels*3
       maxpool2x2    16X108Xbasenum_kernels*3
       =========LAYER4========
       conv   5x5    16x108xbasenum_kernels*4
       maxpool2x2    8X54Xbasenum_kernels*4
       =========LAYER5========
       conv   5x5    8x54xbasenum_kernels*5
       maxpool2x2    4X27Xbasenum_kernels*5
       =========LAYER6========
       conv   2x2    4x27xbasenum_kernels*6
       maxpool2x3    2X9Xbasenum_kernels*6
       =========FCN1========
       fully        (2*9*basenum_kernels*6) x hidden_size
       =========TOP===========
       fully        hidden_size x num_classes
    """
    # preprocess: convert the input img tensor to 4D (required by the conv2d function)
    batch_expanded = tf.expand_dims(batch, axis=3)
    print("batch expanded shape:", batch_expanded.get_shape().as_list())
    # representation learning
    layer1, reg1 = conv2d_layer(batch_expanded, 5,5,basenum_kernels, 2,2,
                                with_relu=True,
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training_placeholder,
                                padding="SAME")
    layer2, reg2 = conv2d_layer(layer1, 5,5,basenum_kernels*2, 2,2,
                                with_relu=True,
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training_placeholder,
                                padding="SAME")
    layer3, reg3 = conv2d_layer(layer2, 5,5,basenum_kernels*3, 2,2,
                                with_relu=True,
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training_placeholder,
                                padding="SAME")
    layer4, reg4 = conv2d_layer(layer3, 5,5,basenum_kernels*4, 2,2,
                                with_relu=True,
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training_placeholder,
                                padding="SAME")
    layer5, reg5 = conv2d_layer(layer4, 5,5,basenum_kernels*5, 2,2,
                                with_relu=True,
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training_placeholder,
                                padding="SAME")
    layer6, reg6 = conv2d_layer(layer5, 2,2,basenum_kernels*6, 2,3,
                                with_relu=True,
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training_placeholder,
                                padding="SAME")
    # top classifier
    conv_out_flat = flatten_batch(layer6)
    fc1, reg7 = fully_connected_layer(conv_out_flat, hidden_size,
                                      with_relu=True, droprate_placeholder=droprate_placeholder)
    logits, reg8 = fully_connected_layer(fc1, num_classes, with_relu=False)
    l2reg = reg1+reg2+reg3+reg4+reg5+reg6+reg7+reg8
    return logits, l2reg





####################################################################################################
### DEFINE TF GRAPH
####################################################################################################

def make_custom_graph(model, chunkshape, num_classes, optimizer_manager, normalize_data=True):
    """
    """
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        optimizer, opt_lrate = optimizer_manager.make_nodes()
        placeholders = {
            "data_placeholder" : tf.placeholder(tf.float32, shape=((None,)+chunkshape), name="data_placeholder"),
            "labels_placeholder" : tf.placeholder(tf.int32, shape=[None], name="labels_placeholder"),
            "l2_rate" : tf.placeholder(tf.float32, name="l2_rate"),
            "dropout_rate" : tf.placeholder(tf.float32, name="dropout_rate"), # placeholder to be able to set it as zero for testing
            "is_training" : tf.placeholder(tf.bool, name="is_training")
        }
        phs = placeholders # alias
        if normalize_data:
            data_batch = tf.nn.l2_normalize(phs["data_placeholder"], 0)
        else:
            data_batch = phs["data_placeholder"]
        logits, l2reg = model(data_batch, num_classes, phs["dropout_rate"], phs["is_training"])
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=phs["labels_placeholder"], logits=logits))
        loss += phs["l2_rate"]*l2reg
        global_step = tf.Variable(0, name="global_step", trainable=False) # to track global step
        minimizer = optimizer.minimize(loss, global_step=global_step)
        predictions = tf.argmax(logits, 1, output_type=tf.int32)
        correct_predictions = tf.equal(predictions, phs["labels_placeholder"])
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        outputs = {"loss":loss, "minimizer":minimizer, "opt_lrate":opt_lrate,
                   "predictions":predictions, "accuracy":accuracy, "logits":logits}
        return g, placeholders, outputs


####################################################################################################
### TF SESSION HELPERS
####################################################################################################

def vote(int_arr):
    """Usage example: vote([1,1,1,1,2,2,2,3,3,3]) returns 1
    """
    return np.argmax(np.bincount(int_arr))

def make_timestamp():
    return '{:%d_%b_%Y_%Hh%Mm%Ss}'.format(datetime.datetime.now())

def make_session_datastring(model, opt_manager, batch_size, chunk_size,
                         l2_0, dropout_0, cv_voting_size, test_voting_size, normalize_chunks,
                         extra_info="", separator="||"):
    s = str(separator)
    if extra_info:
        extra_info = s+extra_info
    return (make_timestamp()+"/"+model.__name__+s+str(opt_manager)+s+"batchsize_"+
            str(batch_size)+s+"chunksize_"+str(chunk_size)+s+"L2_"+str(l2_0)+s+"dropout_"+
            str(dropout_0)+s+"cvvotingsize_"+str(cv_voting_size)+s+"testvotingsize_"+
            str(test_voting_size)+s+"normalizechunks_"+str(normalize_chunks)+extra_info)

def expdecay(step, initval=1e-4, decayratio=3e-7):
    return initval*np.e**(-step*decayratio)


def get_random_batch(data_dict, chunk_size, batch_size):
    """Given a dict of form data_dict[raaga][rec] = (height, width), and two integers representing
       batch size and chunk size, returns a tensor of shape (batch_size, height, chunk_size),
       where raaga, rec, and chunk position are randomly chosen using a uniform distribution,
       and its corresponding list of labels.
       The starting point of the chunk is always between zero and width-chunk_size.
    """
    raagas = [random.choice(data_dict.keys()) for _ in xrange(batch_size)]
    rag_rec = [(r, random.choice(data_dict[r].keys())) for r in raagas]
    rag_rec_chunk = [(raaga, rec, random.randint(0, data_dict[raaga][rec].shape[-1] - chunk_size))
                     for raaga, rec in rag_rec]
    data = np.stack([slice_lastdimension(data_dict[raaga][rec], chunk, chunk+chunk_size)
                       for raaga, rec, chunk in rag_rec_chunk])
    return data.astype(np.float32), raagas

def cut_sample_to_chunks(data, chunk_size, shuffle=True):
    """Given a numpy array of rank R=[...X] and a positive integer smaller than X,
       returns an array of rank R+1 (N, ... chunk_size), where N is the number of chunks
       that could be made by slicing the array along the X dimension (the last one). All chunks will
       be of size chunk_size. Note that if the last chunk is smaller it will be discarded
    """
    width = data.shape[-1]
    num_chunks = width//chunk_size
    if num_chunks<1:
        prompted_warning("in cut_sample_to_chunks (chunk_size="+str(chunk_size) +
                         ") returned non-positive number of chunks for data with shape "+str(data.shape))
    trimmed = slice_lastdimension(data, 0, num_chunks*chunk_size)
    reshaped = np.reshape(trimmed, (num_chunks,)+trimmed.shape[:-1]+(chunk_size,))
    if shuffle: np.random.shuffle(reshaped)
    return reshaped

def get_class_batch(data_dict, clss, chunk_size, batch_size=None):
    """
    """
    if not batch_size:
        batch_size = len(data_dict[clss])
    recs = [random.choice(data_dict[clss].keys()) for _ in xrange(batch_size)]
    rec_chunk = [(rec, random.randint(0, data_dict[clss][rec].shape[1] - chunk_size))
                 for rec in recs]
    data = np.stack([data_dict[clss][rec][:, chunk:chunk+chunk_size] for rec,chunk in rec_chunk])
    return data.astype(np.float32)


class OptimizerManager(object):
    def __init__(self, opt_class, **opt_kwargs):
        self.optimizer_type = opt_class
        self.optimizer_name = opt_class.__name__
        self.optimizer_args = opt_kwargs
        self.pretty_optimizer_args = "("+"|".join(['%s=%s'%(k, v(0)) for k,v in opt_kwargs.iteritems()])+")"
    def make_nodes(self):
        """This can't be in the constructor because the nodes have to be instantiated
           in the 'with' context of their corresponding graph.
        """
        self.placeholders = {arg:tf.placeholder(tf.float32, name=arg) for arg in self.optimizer_args}
        self.optimizer = self.optimizer_type(**self.placeholders)
        lrate = self.optimizer._lr if self.optimizer_type is tf.train.AdamOptimizer else self.optimizer._learning_rate
        return self.optimizer, lrate
    def feed_dict(self, step):
        """for a given training step, returns a feed_dict that can be passed to a TF session
           for training the optimizer of this specific instance.
        """
        return {self.placeholders[arg] : self.optimizer_args[arg](step) for arg in self.optimizer_args}
    def __str__(self):
        return self.optimizer_type.__name__ +self.pretty_optimizer_args


class ConfusionMatrix(object):
    def __init__(self, class_domain, name="", voting_size=None):
        self.matrix = {c:{c:0 for c in class_domain} for c in class_domain}
        self.name = name
        self.voting_info = "(voting size="+str(voting_size)+")" if voting_size else ""
    def add(self, predictions, labels):
        """Given a list of predictions and their respective labels, adds every
           corresponding entry to the confusion matrix.
        """
        for l in labels:
            for p in predictions:
                self.matrix[l][p] += 1
    def __str__(self):
        acc, acc_by_class = self.accuracy()
        classes = sorted(self.matrix.keys())
        short_classes = {c: c[0:8]+"..." if len(c)>8 else c for c in classes}
        prettymatrix = tabulate([[short_classes[c1]]+[self.matrix[c1][c2] for c2 in classes]+
                                 [acc_by_class[c1]]
                                 for c1 in classes],
                                headers=["real(row)|predicted(col)"]+[short_classes[c] for c in classes]+["acc. by class"])
        return ("\n"+self.name+" CONFUSION MATRIX "+self.voting_info+":\n"+prettymatrix+
                "\n"+self.name+" ACCURACY="+str(acc)+"\n")
    def accuracy(self):
        """Returns the total accuracy, and a dictionary with the accuracy per class
        """
        total = 0
        right = 0
        by_class = {c: [0,0] for c in self.matrix}
        acc = float("nan")
        by_class_acc = {c:float("nan") for c in self.matrix}
        for clss, preds in self.matrix.iteritems():
            for pred, n in preds.iteritems():
                total += n
                by_class[clss][1] += n
                if clss==pred:
                    right += n
                    by_class[clss][0] += n
        try:
            acc = float(right)/total
        except ZeroDivisionError:
            pass
        for c,x in by_class.iteritems():
            try:
                by_class_acc[c] = float(x[0])/x[1]
            except ZeroDivisionError:
                pass
        return acc, by_class_acc



class TensorBoardLogger(object):
    def __init__(self, log_dir, log_name, graph):
        self.log_dir = log_dir
        self.log_name = log_name
        self.train_writer = tf.summary.FileWriter(log_dir+log_name+"/train", graph, flush_secs=30)
        self.cv_writer = tf.summary.FileWriter(log_dir+log_name+"/cv", graph, flush_secs=30)
        self.test_writer = tf.summary.FileWriter(log_dir+log_name+"/test", graph, flush_secs=30)
        self.lrate_writer = tf.summary.FileWriter(log_dir+log_name+"/lrate", graph, flush_secs=30)
    def __add_scalars(self, writer, label, acc, loss, step):
        summary = tf.Summary()
        summary.value.add(tag=label+" accuracy", simple_value=acc)
        summary.value.add(tag=label+" loss", simple_value=loss)
        writer.add_summary(summary, step)
    def add_train_scalars(self, acc, loss, lrate, step):
        self.__add_scalars(self.train_writer, "batch", acc, loss, step)
        smm = tf.Summary()
        smm.value.add(tag="learning rate", simple_value=lrate)
        self.lrate_writer.add_summary(smm, step)
    def add_cv_scalars(self, acc, loss, step):
        self.__add_scalars(self.cv_writer, "CV", acc, loss, step)
    def add_test_scalars(self, acc, loss, step):
        self.__add_scalars(self.test_writer, "TEST", acc, loss, step)
    def __add_confmatrix(self, writer, label, conf_matrix, step, color_map="jet", number_color="black"):
        # get clean data from ConfusionMatrix instance and edit main log label
        total_acc, by_class_acc = conf_matrix.accuracy()
        m = conf_matrix.matrix
        classes = sorted(m.keys())
        short_classes = {c: c[0:8]+"..." if len(c)>8 else c for c in classes}
        matrix = np.array([[m[c1][c2] for c2 in classes] for c1 in classes])
        # plot as matrix of colored squares, and "annotate" (fill squares with numbers)
        plt.clf()
        plt.figure()
        plt.imshow(matrix, cmap=color_map)
        for y, c1 in enumerate(classes):
            for x, c2 in enumerate(classes):
                plt.annotate(str(m[c1][c2]), xy=(x, y), color=number_color, fontsize=8,
                             horizontalalignment="center", verticalalignment="center")
        # add titlle, ticks and bar to the plot
        logname1 = self.log_name[:len(self.log_name)//3]
        logname2 = self.log_name[len(self.log_name)//3:2*len(self.log_name)//3]
        logname3 = self.log_name[2*len(self.log_name)//3:]
        label_extra_data = logname1+"\n"+logname2+"\n"+logname3
        plt.title(label+"\n"+label_extra_data+"\n"+
                  "(step="+str(step)+", total acc.="+"%.7f"%total_acc+")")
        plt.colorbar()
        plt.yticks(range(matrix.shape[0]), [short_classes[c]+" (acc="+"%.4f"%by_class_acc[c]+")" for c in classes],
                   fontsize=10, family="DejaVu Sans")
        plt.xticks(range(matrix.shape[1]), [short_classes[c] for c in classes],
                   fontsize=10, family="DejaVu Sans")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        # save plot to buffer in RAM
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches="tight", pad_inches=0.1, dpi=200)
        buf.seek(0)
        # set plot as tensor, pass it to a TF summary and write its result to tensorboard
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        summary = tf.summary.image(label+" "+self.log_name+"(step="+"%010d"%step+
                                   ", total acc.="+"%.7f"%total_acc+")", image, max_outputs=1)
        writer.add_summary(summary.eval(), step)
    def add_cv_confmatrix(self, conf_matrix, step, color_map="jet", number_color="black"):
        self.__add_confmatrix(self.cv_writer, "CV CONFUSION MATRIX", conf_matrix, step, color_map, number_color)
    def add_test_confmatrix(self, conf_matrix, step, color_map="jet", number_color="white"):
        self.__add_confmatrix(self.test_writer, "TEST CONFUSION MATRIX", conf_matrix, step, color_map, number_color)
    def close(self):
        self.train_writer.close()
        self.cv_writer.close()
        self.test_writer.close()



####################################################################################################
### DEFINE TF SESSION
####################################################################################################


def run_training(train_ddict, cv_ddict, test_ddict,
                 model, optimizer_manager,
                 batch_size, chunk_size, l2rate_fn, dropout_fn,
                 log_path,
                 batch_frequency=100, cv_frequency=500, snapshot_frequency=5000,
                 cv_voting_size=1, test_voting_size=1,
                 max_steps=float("inf"),
                 extra_info="",
                 normalize_chunks=True,
                 max_gpu_memory_fraction=1):
     # DATA HOUSEKEEPING: get num of classes, data shape, and create a bijection from its labels to
    # ascending ints starting by zero
    num_classes = len(train_ddict)
    data_shape = train_ddict.values()[0].values()[0].shape
    chunk_shape = (data_shape[0], chunk_size) if len(data_shape)==2 else (chunk_size,)
    class2int = {c:i for i, c in enumerate(train_ddict)}
    int2class = {v:k for k,v in class2int.iteritems()}
    # make graph
    g, graph_placeholders, graph_outputs = make_custom_graph(model, chunk_shape,
                                                             num_classes, optimizer_manager,
                                                             normalize_data=normalize_chunks)
    # run graph
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = max_gpu_memory_fraction
    with tf.Session(graph=g, config=sess_config) as sess:
        # session initialization
        sess.run(tf.global_variables_initializer())
        print("\nTF session initialized.\nMODEL:", model.__name__,
              "\nOPTIMIZER:", optimizer_manager,
              "\nBATCH SIZE:", batch_size,
              "\nCHUNK SHAPE:", chunk_shape,
              "\nL2 REGULARIZATION FACTOR:", l2rate_fn(0),
              "\nDROPOUT:", dropout_fn(0),
              "\nCV VOTING SIZE:", cv_voting_size,
              "\nTEST VOTING SIZE:", test_voting_size,
              "\nNORMALIZE CHUNKS", normalize_chunks)
        # tensorboard logging
        sess_datastring = make_session_datastring(model, optimizer_manager, batch_size, chunk_size,
                                                  l2rate_fn(0), dropout_fn(0), cv_voting_size,
                                                  test_voting_size, normalize_chunks, extra_info)
        log_dir = log_path + sess_datastring
        logger = TensorBoardLogger(log_path, sess_datastring, g)
        # start optimization
        step = 0
        while step<max_steps:
            step += 1
            # TRAINING
            train_data_batch, train_lbl_batch = get_random_batch(train_ddict, chunk_size, batch_size)
            train_lbl_batch = [class2int[lbl] for lbl in train_lbl_batch]
            train_feed = {graph_placeholders["data_placeholder"]:train_data_batch,
                          graph_placeholders["labels_placeholder"]:train_lbl_batch,
                          graph_placeholders["l2_rate"]: l2rate_fn(step),
                          graph_placeholders["dropout_rate"]:dropout_fn(step),
                          graph_placeholders["is_training"]:True}
            train_feed.update(optimizer_manager.feed_dict(step))
            _, lrate = sess.run([graph_outputs["minimizer"],
                                 graph_outputs["opt_lrate"]], feed_dict=train_feed)
            # plot training
            if step%batch_frequency == 0:
                plot_feed = {graph_placeholders["data_placeholder"]:train_data_batch,
                             graph_placeholders["labels_placeholder"]:train_lbl_batch,
                             graph_placeholders["l2_rate"]: l2rate_fn(step),
                             graph_placeholders["dropout_rate"]:1.0,
                             graph_placeholders["is_training"]:False}
                acc, lss = sess.run([graph_outputs["accuracy"],graph_outputs["loss"]], feed_dict=plot_feed)
                print("[TRAINING]","\tstep = "+str(step)+"/"+str(max_steps), "\tbatch_acc =", acc, "\tbatch_loss =", lss)
                logger.add_train_scalars(acc, lss, lrate, step)
            # CROSS VALIDATION
            if step%cv_frequency == 0:
                confmatrix = ConfusionMatrix(cv_ddict.keys(), "CV", voting_size=cv_voting_size)
                cv_total_loss = 0
                # split each CV sample into chunks and make a single batch with them:
                for ccc in cv_ddict:
                    for _, data in cv_ddict[ccc].iteritems():
                        cv_sample_data = cut_sample_to_chunks(data, chunk_size, shuffle=True)
                        cv_sample_len = len(cv_sample_data)
                        # pass the sample chunks to TF
                        for ii in xrange(0, cv_sample_len, cv_voting_size):
                            max_ii = min(ii+cv_voting_size, cv_sample_len) # avoids crash in last chunk
                            if(max_ii-ii<cv_voting_size):
                                print("CV warning: voting among", max_ii-ii, "elements, whereas",
                                      "voting size was", cv_voting_size,
                                      "(data_length, chunksize) =", (data.shape[1], chunk_size))
                            cv_data_batch = cv_sample_data[ii:max_ii]
                            cv_labels_batch = [class2int[ccc] for _ in xrange(len(cv_data_batch))]
                            cv_feed = {graph_placeholders["data_placeholder"]:cv_data_batch,
                                       graph_placeholders["labels_placeholder"]:cv_labels_batch,
                                       graph_placeholders["l2_rate"]: l2rate_fn(step),
                                       graph_placeholders["dropout_rate"]:1.0,
                                       graph_placeholders["is_training"]:False}
                            cv_preds, cv_lss = sess.run([graph_outputs["predictions"],
                                                         graph_outputs["loss"]], feed_dict=cv_feed)
                            cv_total_loss += cv_lss
                            confmatrix.add([int2class[vote(cv_preds)]], [ccc])
                print(confmatrix, "CV LOSS = ", cv_total_loss, sep="")
                logger.add_cv_scalars(confmatrix.accuracy()[0], cv_total_loss, step)
                logger.add_cv_confmatrix(confmatrix, step)
        # TESTING

        confmatrix = ConfusionMatrix(test_ddict.keys(), "TEST", voting_size=test_voting_size)
        test_total_loss = 0
        # split each TEST sample into chunks and make a single batch with them:
        for ccc in test_ddict:
            for _, data in test_ddict[ccc].iteritems():
                test_sample_data = cut_sample_to_chunks(data, chunk_size, shuffle=True)
                test_sample_len = len(test_sample_data)
                # pass the sample chunks to TF
                for ii in xrange(0, test_sample_len, test_voting_size):
                    max_ii = min(ii+test_voting_size, test_sample_len) # avoids crash in last chunk
                    if(max_ii-ii<test_voting_size):
                        print("TEST warning: voting among", max_ii-ii, "elements, whereas",
                              "voting size was", test_voting_size,
                              "(data_length, chunksize) =", (data.shape[1], chunk_size))
                    test_data_batch = test_sample_data[ii:max_ii]
                    test_labels_batch = [class2int[ccc] for _ in xrange(len(test_data_batch))]
                    test_feed = {graph_placeholders["data_placeholder"]:test_data_batch,
                               graph_placeholders["labels_placeholder"]:test_labels_batch,
                               graph_placeholders["l2_rate"]: l2rate_fn(step),
                               graph_placeholders["dropout_rate"]:1.0,
                               graph_placeholders["is_training"]:False}
                    test_preds, test_lss = sess.run([graph_outputs["predictions"],
                                                 graph_outputs["loss"]], feed_dict=test_feed)
                    test_total_loss += test_lss
                    confmatrix.add([int2class[vote(test_preds)]], [ccc])
        print(confmatrix, "TEST LOSS = ", test_total_loss, sep="")
        logger.add_test_scalars(confmatrix.accuracy()[0], test_total_loss, step)
        logger.add_test_confmatrix(confmatrix, step)
        logger.close()






####################################################################################################
### LOAD DATA AND RUN TF MODEL
####################################################################################################

def load_mnist_as_ddicts():
    """
    """
    # load the dataset from TF dependencies
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    # extract the tensors and print their shape
    train_labels = mnist.train._labels # 55000
    train_images = mnist.train._images.reshape((-1, 28, 28))
    cv_labels = mnist.validation._labels # 5000
    cv_images = mnist.validation._images.reshape((-1, 28, 28))
    test_labels = mnist.test._labels # 10000
    test_images = mnist.test._images.reshape((-1, 28, 28))
    print(train_labels.shape, train_images.shape, cv_labels.shape, cv_images.shape,
          test_labels.shape, test_images.shape)
    # store them as dicts of dicts
    train_ddict = {str(i):{} for i in xrange(10)} # this are still empty!
    cv_ddict = {str(i):{} for i in xrange(10)}
    test_ddict = {str(i):{} for i in xrange(10)}
    for i in xrange(len(train_labels)):
        train_ddict[str(train_labels[i])][str(i)] = train_images[i]
    for i in xrange(len(cv_labels)):
        cv_ddict[str(cv_labels[i])][str(i)] = cv_images[i]
    for i in xrange(len(test_labels)):
        test_ddict[str(test_labels[i])][str(i)] = test_images[i]
    # return them
    return train_ddict, cv_ddict, test_ddict


### MNIST
LOG_PATH = "../tensorboard_logs/"
TRAIN_MNIST_DDICT, CV_MNIST_DDICT, TEST_MNIST_DDICT = load_mnist_as_ddicts()
BATCHSIZE = 200
CHUNKSIZE = 28
# OPTMANAGER = OptimizerManager(tf.train.MomentumOptimizer, learning_rate=lambda x: 1e-4, momentum=lambda x:0.5)
OPTMANAGER = OptimizerManager(tf.train.AdamOptimizer, learning_rate=lambda x: 1e-4)
SELECTED_MODEL = mnist_model_conv
L2REG_FN = lambda x:  3e-07
DROPOUT_FN = lambda x: 1

run_training(TRAIN_MNIST_DDICT, CV_MNIST_DDICT, TEST_MNIST_DDICT,
             SELECTED_MODEL, OPTMANAGER,
             BATCHSIZE, CHUNKSIZE, L2REG_FN, DROPOUT_FN,
             LOG_PATH,
             batch_frequency=100, cv_frequency=500, snapshot_frequency=5000,
             cv_voting_size=1, test_voting_size=1,
             max_steps=1000,
             normalize_chunks=False)



input("stop after mnist!")









### CARNATIC
def carnatic_model_basic_base2(batch, num_classes, droprate_placeholder, is_training):
    return carnatic_model_basic(batch, num_classes, droprate_placeholder, is_training, basenum_kernels=2)
def carnatic_model_basic_base4(batch, num_classes, droprate_placeholder, is_training):
    return carnatic_model_basic(batch, num_classes, droprate_placeholder, is_training, basenum_kernels=4)
def carnatic_model_basic_base6(batch, num_classes, droprate_placeholder, is_training):
    return carnatic_model_basic(batch, num_classes, droprate_placeholder, is_training, basenum_kernels=6)
def carnatic_model_basic_base8(batch, num_classes, droprate_placeholder, is_training):
    return carnatic_model_basic(batch, num_classes, droprate_placeholder, is_training, basenum_kernels=8)
def carnatic_model_basic_base10(batch, num_classes, droprate_placeholder, is_training):
    return carnatic_model_basic(batch, num_classes, droprate_placeholder, is_training, basenum_kernels=10)
def carnatic_model_basic_base12(batch, num_classes, droprate_placeholder, is_training):
    return carnatic_model_basic(batch, num_classes, droprate_placeholder, is_training, basenum_kernels=12)
def carnatic_model_basic_base14(batch, num_classes, droprate_placeholder, is_training):
    return carnatic_model_basic(batch, num_classes, droprate_placeholder, is_training, basenum_kernels=14)
def carnatic_model_basic_base16(batch, num_classes, droprate_placeholder, is_training):
    return carnatic_model_basic(batch, num_classes, droprate_placeholder, is_training, basenum_kernels=16)





def cqt5(batch, num_classes, droprate_placeholder, is_training, basenum_kernels=16):
    """A simple fully conv network:
       input:       180X860
       reshaped to  180x860x1
       =========LAYER1========
       conv   2x2    179x859xbasenum_kernels
       maxpool2x2     89x429xbasenum_kernels
       =========LAYER2========
       conv   2x2    88x428xbasenum_kernels*4
       maxpool2x2    44x214Xbasenum_kernels*4
       =========LAYER3========
       conv   8x40   37x175xbasenum_kernels*16
       maxpool1x4    37X43Xbasenum_kernels*16
       =========LAYER4========
       conv   2x2    36x42xbasenum_kernels*20
       maxpool3x3    12X14Xbasenum_kernels*20
       =========TOP===========
       conv+dropout 12x14  1x1xbasenum_kernels*20
       conv         1x1    1x1xnum_classes
    """
    # preprocess: convert the input img tensor to 4D (required by the conv2d function)
    batch_expanded = tf.expand_dims(batch, axis=3)
    print("batch expanded shape:", batch_expanded.get_shape().as_list())
    # representation learning
    layer1, reg1 = conv2d_layer(batch_expanded, 2,2,basenum_kernels, 2,2, with_relu=True, padding="VALID")
    layer2, reg2 = conv2d_layer(layer1,   2,2,basenum_kernels*4,  2,2, with_relu=True, padding="VALID")
    layer3, reg3 = conv2d_layer(layer2, 8,40,basenum_kernels*16,  1,4, with_relu=True, padding="VALID")
    layer4, reg4 = conv2d_layer(layer3, 2,2,basenum_kernels*20,  3,3, with_relu=True, padding="VALID")
    # spaghetti
    layer5, reg5 = conv2d_layer(layer4, 12, 14, basenum_kernels*20, 1,1, with_relu=True,
                                padding="VALID", droprate_placeholder=droprate_placeholder)
    # top classifier
    logits, reg6 = conv2d_layer(layer5, 1, 1, num_classes, 1, 1, with_relu=False, padding="VALID")
    logits = tf.squeeze(logits)
    l2reg = reg1+reg2+reg3+reg4+reg5+reg6
    return logits, l2reg



def fcn4(batch, num_classes, droprate_placeholder, is_training, basenum_kernels=16):
    """A simple fully conv network:
       input         180X860x1
       =========LAYER1========
       conv   3x3    178x858xbasenum_kernels
       maxpool2x4     89x214xbasenum_kernels
       =========LAYER2========
       conv   3x3    87x212xbasenum_kernels*3
       maxpool4x5    21x42xbasenum_kernels*3
       =========LAYER3========
       conv   3x3    19x40xbasenum_kernels*6
       maxpool3x5    6X8Xbasenum_kernels*6
       =========LAYER4=======
       conv   3x3    4x6xbasenum_kernels*16
       maxpool4x6    1X1Xbasenum_kernels*16
       =========TOP===========
       conv         1x1    1x1xnum_classes
    """
    # preprocess: convert the input img tensor to 4D (required by the conv2d function)
    batch_size = tf.shape(batch)[0]
    batch_expanded = tf.expand_dims(batch, axis=3)
    print("batch expanded shape:", batch_expanded.get_shape().as_list())
    # representation learning
    layer1, reg1 = conv2d_layer(batch_expanded, 3,3,basenum_kernels, 2,4, with_relu=True,
                                droprate_placeholder=None, padding="VALID",
                                with_batchnorm=True, is_training_placeholder=is_training)
    layer2, reg2 = conv2d_layer(layer1, 3,3,basenum_kernels*3, 4,5, with_relu=True,
                                droprate_placeholder=None, padding="VALID",
                                with_batchnorm=True, is_training_placeholder=is_training)
    layer3, reg3 = conv2d_layer(layer2, 3,3,basenum_kernels*6, 3,5, with_relu=True,
                                droprate_placeholder=None, padding="VALID",
                                with_batchnorm=True, is_training_placeholder=is_training)
    layer4, reg4 = conv2d_layer(layer3, 3,3,basenum_kernels*16, 4,6, with_relu=True,
                                droprate_placeholder=droprate_placeholder, padding="VALID",
                                with_batchnorm=True, is_training_placeholder=is_training)
    # top classifier
    logits, reg5 = conv2d_layer(layer4, 1, 1, num_classes, 1, 1, with_relu=False)
    logits = tf.reshape(logits, [batch_size, num_classes])
    l2reg = reg1+reg2+reg3+reg4+reg5
    return logits, l2reg


def fcn4_180(batch, num_classes, droprate_placeholder, is_training, basenum_kernels=64,
             with_batchnorm=False):
    """A simple fully conv network:
       input         180X180x1
       =========LAYER1========
       conv   3x3    178x858xbasenum_kernels
       maxpool3x3     59x59xbasenum_kernels
       =========LAYER2========
       conv   3x3    57x57xbasenum_kernels*3
       maxpool3x3    19x19xbasenum_kernels*3
       =========LAYER3========
       conv   3x3    17x17xbasenum_kernels*6
       maxpool3x3    5 X 5Xbasenum_kernels*6
       =========LAYER4=======
       conv   3x3    3x3xbasenum_kernels*16
       maxpool3x3    1X1Xbasenum_kernels*16
       =========TOP===========
       conv         1x1    1x1xnum_classes
    """
    # preprocess: convert the input img tensor to 4D (required by the conv2d function)
    batch_size = tf.shape(batch)[0]
    batch_expanded = tf.expand_dims(batch, axis=3)
    print("batch expanded shape:", batch_expanded.get_shape().as_list())
    # representation learning
    layer1, reg1 = conv2d_layer(batch_expanded, 3,3,basenum_kernels, 3,3, with_relu=True,
                                droprate_placeholder=droprate_placeholder, padding="VALID",
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training)
    layer2, reg2 = conv2d_layer(layer1, 3,3,basenum_kernels*3, 3,3, with_relu=True,
                                droprate_placeholder=droprate_placeholder, padding="VALID",
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training)
    layer3, reg3 = conv2d_layer(layer2, 3,3,basenum_kernels*6, 3,3, with_relu=True,
                                droprate_placeholder=droprate_placeholder, padding="VALID",
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training)
    layer4, reg4 = conv2d_layer(layer3, 3,3,basenum_kernels*16, 3,3, with_relu=True,
                                droprate_placeholder=droprate_placeholder, padding="VALID",
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training)
    # top classifier
    logits, reg5 = conv2d_layer(layer4, 1, 1, num_classes, 1, 1, with_relu=False)
    logits = tf.reshape(logits, [batch_size, num_classes])
    l2reg = reg1+reg2+reg3+reg4+reg5
    return logits, l2reg


def fcn4_128(batch, num_classes, droprate_placeholder, is_training, basenum_kernels=200,
             with_batchnorm=False):
    """A simple fully conv network:
       input         128X128x1
       =========LAYER1========
       conv   3x3    126x126xbasenum_kernels
       maxpool3x3     42x42xbasenum_kernels
       =========LAYER2========
       conv   3x3    40x40xbasenum_kernels*3
       maxpool2x2    20x20xbasenum_kernels*3
       =========LAYER3========
       conv   3x3    18x18xbasenum_kernels*6
       maxpool3x3     6X6Xbasenum_kernels*6
       =========LAYER4=======
       conv   3x3    4x4xbasenum_kernels*9
       maxpool2x2    2X2Xbasenum_kernels*9
       =========TOP===========
       conv  2x2      2x2xnum_classes
       maxpool2x2     1x1xnum_classes
    """
    # preprocess: convert the input img tensor to 4D (required by the conv2d function)
    batch_size = tf.shape(batch)[0]
    batch_expanded = tf.expand_dims(batch, axis=3)
    print("batch expanded shape:", batch_expanded.get_shape().as_list())
    # representation learning
    layer1, reg1 = conv2d_layer(batch_expanded, 3,3,basenum_kernels, 3,3, with_relu=True,
                                droprate_placeholder=droprate_placeholder, padding="VALID",
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training)
    layer2, reg2 = conv2d_layer(layer1, 3,3,basenum_kernels*3, 2,2, with_relu=True,
                                droprate_placeholder=droprate_placeholder, padding="VALID",
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training)
    layer3, reg3 = conv2d_layer(layer2, 3,3,basenum_kernels*6, 3,3, with_relu=True,
                                droprate_placeholder=droprate_placeholder, padding="VALID",
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training)
    layer4, reg4 = conv2d_layer(layer3, 3,3,basenum_kernels*9, 2,2, with_relu=True,
                                droprate_placeholder=droprate_placeholder, padding="VALID",
                                with_batchnorm=with_batchnorm, is_training_placeholder=is_training)
    # top classifier
    logits, reg5 = conv2d_layer(layer4, 2, 2, num_classes, 2, 2, with_relu=False)
    logits = tf.reshape(logits, [batch_size, num_classes])
    l2reg = reg1+reg2+reg3+reg4+reg5
    return logits, l2reg


#################################################

# get data
LOG_PATH = "../tensorboard_logs/"
MEL_H5PATH = "../datasets/melgrams_22050_4096_4.h5"
CQT_H5PATH = "../datasets/cqts_22050_4096_4.h5"
_, GULATI_RAAGAS = load_dunya_raaga_lists()
CLASS_SET = [GULATI_RAAGAS[x][0] for x in [0,2]] # Todi and Kalyaani
# TRAIN_CARNATIC_DDICT, TEST_CARNATIC_DDICT = load_carnatic_as_ddicts(MEL_H5PATH, CLASS_SET)
TRAIN_CARNATIC_DDICT, TEST_CARNATIC_DDICT = load_gulati_wavdata(WAV_22050_H5PATH, CLASS_SET, 0.5)



# #  bsize, chsize,     optimizer                        opt args,                   l2,        dropout            model
# #this tended to say always class1
# l1 = [40, 20*43, tf.train.GradientDescentOptimizer, {"learning_rate":lambda x:1e-5},lambda x:0, lambda x: 1, carnatic_model_basic_base16]
# # this rises batch acc about 0.2 every 5k steps, but starts overfitting a lot after 7k
# l2 = [40, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:1e-10, lambda x: 1, carnatic_model_basic_base16]
# # this also overfits, more l2!
# l3 = [40, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:1e-9, lambda x: 1, carnatic_model_basic_base16]
# # this overtfits even more than l3: train goes up normally, but test cost skyrokets and acc. blocks bc it says that everything is class 2
# l4 = [40, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:1e-8, lambda x: 1, carnatic_model_basic_base16]
# # lets see with dropout
# l5 = [40, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:0, lambda x: 0.9, carnatic_model_basic_base16]
# l6 = [40, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:0, lambda x: 0.8, carnatic_model_basic_base16]
# l7 = [40, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:0, lambda x: 0.7, carnatic_model_basic_base16]
# l8 = [40, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:0, lambda x: 0.6, carnatic_model_basic_base16]
# # actually, more dropout overfits too (l5-l8)... weird!!. The CV cost skyrockets and the batch acc rises fast too. So lets reduce the model and increase batchsize
# l9  = [60, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:1e-9, lambda x: 1, carnatic_model_basic_base2]
# l10  = [60, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:1e-9, lambda x: 1, carnatic_model_basic_base4]
# l11  = [60, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:1e-9, lambda x: 1, carnatic_model_basic_base6]
# l12  = [60, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:1e-9, lambda x: 1, carnatic_model_basic_base8]
# l13  = [60, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:1e-9, lambda x: 1, carnatic_model_basic_base10]
# l14  = [60, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:1e-9, lambda x: 1, carnatic_model_basic_base12]
# l15  = [60, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:1e-9, lambda x: 1, carnatic_model_basic_base14]
# # confmatrix were more distributed, but still cv barely above random.
# # At this point i would conclude that the mnist-based models arent good for this data.
# # I will try to go for the CQT, bigger kernels (with valid padding) and less layers:
# l16 = [10, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:0, lambda x: 0.5, fcn4]
# l17 = [10, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:0, lambda x: 0.7, fcn4]
# l18 = [10, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:0, lambda x: 0.9, fcn4]
# l19 = [10, 20*43, tf.train.AdamOptimizer, {"learning_rate":lambda x:1e-5},lambda x:0, lambda x: 1, fcn4]
# all this networks cant even get the training data, even with basenum=128. So I removed dropout and printed now the alpha to see whats happening
# l20= [30, 128, tf.train.MomentumOptimizer, {"learning_rate":lambda x: expdecay(x, initval=1e-7, decayratio=1e-4), "momentum": lambda x: 0},lambda x:0, lambda x: 1, fcn4_128]
# l21= [30, 128, tf.train.MomentumOptimizer, {"learning_rate":lambda x: expdecay(x, initval=1e-7, decayratio=1e-4), "momentum": lambda x: 0},lambda x:0, lambda x: 0.95, fcn4_128]
# l22= [30, 128, tf.train.MomentumOptimizer, {"learning_rate":lambda x: expdecay(x, initval=1e-7, decayratio=1e-4), "momentum": lambda x: 0},lambda x:0, lambda x: 0.8, fcn4_128]
# l23= [30, 128, tf.train.MomentumOptimizer, {"learning_rate":lambda x: expdecay(x, initval=1e-7, decayratio=1e-4), "momentum": lambda x: 0},lambda x:0, lambda x: 0.5, fcn4_128]
# none of them learned anything, even on the training data. switching to 1D conv, to discard preproc. mistakes
l100= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.9, wav_simple_model]
l101= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.8, wav_simple_model]
l102= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.7, wav_simple_model]
l103= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.6, wav_simple_model]
l104= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.5, wav_simple_model]
l105= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:1e-5, lambda x: 1, wav_simple_model]
l106= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:1e-6, lambda x: 1, wav_simple_model]
l107= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:1e-7, lambda x: 1, wav_simple_model]
l108= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:1e-8, lambda x: 1, wav_simple_model]
l109= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:1e-9, lambda x: 1, wav_simple_model]
# all of them overfitted clearly. increase reg and SET DROPOUT EVERY LAYER.
l110= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:1e-4, lambda x: 1, wav_simple_model]
l111= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:1e-3, lambda x: 1, wav_simple_model]
l112= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:1e-2, lambda x: 1, wav_simple_model]
l113= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:3e-2, lambda x: 1, wav_simple_model]
l114= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:1e-1, lambda x: 1, wav_simple_model]
l115= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.9, wav_simple_model]
l116= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.8, wav_simple_model]
l117= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.7, wav_simple_model]
l118= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.6, wav_simple_model]
l119= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.5, wav_simple_model]
# drop below 0.9 was too much. finetune
l120= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.91, wav_simple_model]
l121= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.92, wav_simple_model]
l122= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.93, wav_simple_model]
l123= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.94, wav_simple_model]
l124= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.95, wav_simple_model]
l125= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.96, wav_simple_model]
l126= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.97, wav_simple_model]
l127= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.98, wav_simple_model]
l128= [10, 5*22050, tf.train.AdamOptimizer, {"learning_rate":lambda x: 1e-4},lambda x:0, lambda x: 0.99, wav_simple_model]
# now both l2 and DO:
# now reduce model size

for hyperpars in [l121, l122, l123, l124, l125, l126, l127, l128]:
    # adjust hyperparameters
    BATCHSIZE = hyperpars[0]
    CHUNKSIZE = hyperpars[1] # 20 secs is approx  20*43=860
    OPTMANAGER = OptimizerManager(hyperpars[2], **hyperpars[3])
    L2REG_FN = hyperpars[4] #1e-10
    DROPOUT_FN = hyperpars[5]
    SELECTED_MODEL = hyperpars[6]
    run_training(TRAIN_CARNATIC_DDICT, TEST_CARNATIC_DDICT, TEST_CARNATIC_DDICT,
                 SELECTED_MODEL, OPTMANAGER,
                 BATCHSIZE, CHUNKSIZE, L2REG_FN, DROPOUT_FN,
                 LOG_PATH,
                 batch_frequency=50, cv_frequency=2000, snapshot_frequency=5000,
                 cv_voting_size=1, test_voting_size=1,
                 max_steps=50000,
                 normalize_chunks=False)
