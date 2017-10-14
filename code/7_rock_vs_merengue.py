"""as described in https://www.tensorflow.org/tutorials/deep_cnn
   dividing the code in 3 sections (inputs, prediction, training) makes the code most reusable.
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from random import shuffle
import scipy.io.wavfile as pywav
import h5py
import os
import sys


################################################################################
### WAV CHUNKS PREPROCESSING:
### 1) import the wav songs and separate them into training and test datasets
### 2) cut them into 2s chunks (truncate rest) and put into np arrays
### 3) make labels, merge rock&merengue, shuffle and export
################################################################################

ROCK_WAV_IMPORT_DIR = "../datasets/rock/"
MERENGUE_WAV_IMPORT_DIR = "../datasets/merengue/"
DATASET_WAVCHUNKS_FILEPATH = "../datasets/rock_vs_merengue_ds_wavchunks.h5"
SAMPLE_RATE = 22050
CHUNK_SIZE = 2 # in seconds


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

def cutdown_wavlist(wavlist):
    arr = np.concatenate(wavlist).reshape(1, -1, 1)
    samples_per_chunk = int(CHUNK_SIZE*SAMPLE_RATE)
    num_chunks = arr.shape[1]/samples_per_chunk
    return arr[:,0:(num_chunks*samples_per_chunk),:].reshape(num_chunks, samples_per_chunk, 1)

def shuffle_sync_in_place(data, labels):
    if data.shape[0]!=labels.shape[0]:
        raise Exception("~shuffle_sync_in_place: data and labels must have same length!")
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(labels)

def save_dataset_dict(path, dataset):
    with h5py.File(path) as hf:
        for k,v in dataset.items():
            hf.create_dataset(k, data=v, compression="gzip", compression_opts=9)

def load_dataset_dict(path):
    with h5py.File(path, "r") as hf:
        return {k:v.value for k,v in hf.items()}

def make_wavchunks_dataset():
    # load wav songs
    rock_pcm = load_all_wav_files(ROCK_WAV_IMPORT_DIR, SAMPLE_RATE).values()
    merengue_pcm = load_all_wav_files(MERENGUE_WAV_IMPORT_DIR, SAMPLE_RATE).values()
    # split them into training and testing, and cut down into chunks
    l1, l2 = len(rock_pcm), len(merengue_pcm)
    r_train, r_test = cutdown_wavlist(rock_pcm[0:l1/2]), cutdown_wavlist(rock_pcm[l1/2:])
    m_train, m_test = cutdown_wavlist(merengue_pcm[0:l1/2]), cutdown_wavlist(merengue_pcm[l1/2:])
    del rock_pcm, merengue_pcm #no longer needed
    # shuffle, trim overrepresented sets to have balanced classes, and merge r+m into train and test
    [np.random.shuffle(x) for x in (r_train, r_test, m_train, m_test)]
    train_len = min(r_train.shape[0], m_train.shape[0])
    test_len = min(r_test.shape[0], m_test.shape[0])
    train_data = np.concatenate([r_train[0:train_len], m_train[0:train_len]])
    test_data = np.concatenate([r_test[0:test_len], m_test[0:test_len]])
    # make labels (1 for rock, 0 for merengue), and shuffle again
    train_labels = np.concatenate([np.ones(train_len), np.zeros(train_len)])
    test_labels = np.concatenate([np.ones(test_len), np.zeros(test_len)])
    shuffle_sync_in_place(train_data, train_labels)
    shuffle_sync_in_place(test_data, test_labels)
    # return as dictionary
    return {"train_data":train_data, "train_labels":train_labels, "test_data":test_data,
            "test_labels":test_labels}

def make_and_save_wavchunks():
    save_dataset_dict(DATASET_WAVCHUNKS_FILEPATH, make_wavchunks_dataset())



# make_and_save_wavchunks()
# DATASET = load_dataset_dict(DATASET_WAVCHUNKS_FILEPATH)
# print(DATASET["test_data"].shape)
# input("stop!!")


################################################################################
### STFT CHUNKS PREPROCESSING:
### 1) import the wav chunks and calculate STFTs
################################################################################
FFT_SIZE = 512
OVERLAP_FACTOR = 0.5
DATASET_STFTS_FILEPATH = "../datasets/rock_vs_merengue_ds_stfts.h5"


def calculate_stft(data_array, fft_size=FFT_SIZE, overlap_factor=OVERLAP_FACTOR):
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


def make_stfts_dataset():
    ds = make_wavchunks_dataset()
    train_stft = np.array([calculate_stft(x) for x in ds["train_data"][:,:,0]])
    test_stft = np.array([calculate_stft(x) for x in ds["test_data"][:,:,0]])
    ds["train_data"] = np.expand_dims(train_stft, 3)
    ds["test_data"] = np.expand_dims(test_stft, 3)
    return ds

def make_and_save_stfts():
    save_dataset_dict(DATASET_STFTS_FILEPATH, make_stfts_dataset())


# make_and_save_stfts()
# DATASET = load_dataset_dict(DATASET_STFTS_FILEPATH)
# print(DATASET["test_data"].shape)
# input("stop!!!")



################################################################################
### DEFINE TF MODELS
################################################################################


# weights with small noise for symmetry breaking and close to zero to avoid zero-gradients
weight_variable = lambda shape, stddev=0.1: tf.Variable(tf.truncated_normal(shape, stddev=stddev))
# bias greater than zero to avoid dead neurons, since using ReLU
bias_variable = lambda shape: tf.Variable(tf.constant(0.1, shape=[shape]))
# aliases
relu = tf.nn.relu
conv1d = tf.nn.conv1d
conv2d = tf.nn.conv2d
max_pool = tf.nn.max_pool
matmul = tf.matmul
dropout = tf.nn.dropout
l2loss = tf.nn.l2_loss

def basic_model(batch, hidden_size=512):
    # fully_hidden
    W1 = weight_variable([batch.get_shape().as_list()[1], hidden_size])
    b1 = bias_variable(hidden_size)

    out1 = relu(matmul(batch[:,:,0], W1)+b1)
    # logits
    W2 = weight_variable([hidden_size, 2])
    b2 = bias_variable(2)
    return tf.matmul(out1, W2)+b2

def model_conv1d(batch, droprate_placeholder, hidden_size=128):
    """this got above 81% with alpha=1e-4, dropout=0.5, lambda=0.3, batch=100
    """
    # conv layer1
    W = weight_variable([10, 1, 4])
    b = bias_variable(4)
    conv = relu(conv1d(batch, W, stride=2, padding="VALID")+b)
    l2reg = l2loss(W)
    # conv layer 2
    W = weight_variable([20, 4, 8])
    b = bias_variable(8)
    conv = relu(conv1d(conv, W, stride=3, padding="VALID")+b)
    l2reg += l2loss(W)
    # conv layer 3
    W = weight_variable([30, 8, 16])
    b = bias_variable(16)
    conv = relu(conv1d(conv, W, stride=5, padding="VALID")+b)
    l2reg += l2loss(W)
    # conv layer 4
    W = weight_variable([40, 16, 24])
    b = bias_variable(24)
    conv = relu(conv1d(conv, W, stride=8, padding="VALID")+b)
    l2reg += l2loss(W)
    # conv layer 5
    W = weight_variable([50, 24, 32])
    b = bias_variable(32)
    conv = relu(conv1d(conv, W, stride=13, padding="VALID")+b)
    l2reg += l2loss(W)
    # output of CNN: chunks of length 252 and depth 16. Flatten!
    shape = conv.get_shape().as_list()
    flatsize = shape[1]*shape[2]
    conv_flat = tf.reshape(conv, [tf.shape(conv)[0], flatsize])
    # fully connected hidden layer:
    W = weight_variable([flatsize, hidden_size])
    b = bias_variable(hidden_size)
    hidden = relu(matmul(conv_flat, W)+b)
    l2reg += l2loss(W)
    # dropout&logits
    W = weight_variable([hidden_size, 2])
    b = bias_variable(2)
    l2reg += l2loss(W)
    return tf.matmul(dropout(hidden, droprate_placeholder), W)+b, l2reg



def model_conv2d(batch, droprate_placeholder, hidden_size=24):
    # conv layer1
    W = weight_variable([512, 10, 1, 8])
    b = bias_variable(8)
    conv = relu(conv2d(batch, W, strides=[1,1,5,1], padding="VALID")+b)
    #conv = max_pool(conv, ksize([1,4,1,1], strides=[1,4,1,1], padding="SAME"))
    l2reg = l2loss(W)
    # conv layer2
    W = weight_variable([1, 15, 8, 16])
    b = bias_variable(16)
    conv = relu(conv2d(conv, W, strides=[1,1,10,1], padding="VALID")+b)
    l2reg += l2loss(W)
    # # conv layer3
    # W = weight_variable([1, 15, 6, 8])
    # b = bias_variable(8)
    # conv = relu(conv2d(conv, W, strides=[1,1,1,1], padding="VALID")+b)
    # l2reg += l2loss(W)
    # # conv layer4
    # W = weight_variable([1, 20, 8, 12])
    # b = bias_variable(12)
    # conv = relu(conv2d(conv, W, strides=[1,1,1,1], padding="VALID")+b)
    # l2reg += l2loss(W)
    # output of CNN:
    shape = conv.get_shape().as_list()
    flatsize = shape[1]*shape[2]*shape[3]
    conv_flat = tf.reshape(conv, [tf.shape(conv)[0], flatsize])
    # fully connected hidden layer
    W = weight_variable([flatsize, hidden_size])
    b = bias_variable(hidden_size)
    hidden = relu(matmul(conv_flat, W)+b)
    l2reg += l2loss(W)
    # dropout&logits
    W = weight_variable([hidden_size, 2])
    b = bias_variable(2)
    l2reg += l2loss(W)
    return tf.matmul(dropout(hidden, droprate_placeholder), W)+b, l2reg



################################################################################
### RUN TF GRAPH
################################################################################

#
BATCH_SIZE = 15 # 20
ALPHA= 1e-4 # 1e-4
LAMBDA= 0 # 0.2
DROPOUT_RATE= 1 # 1
DS_PATH = DATASET_STFTS_FILEPATH # DATASET_WAVCHUNKS_FILEPATH #

# prevent loading DS multiple times
try:
    DATASET
except:
    DATASET = load_dataset_dict(DS_PATH)
# convenience globals
TRAIN_D = DATASET["train_data"]
TEST_D = DATASET["test_data"]
TRAIN_L = DATASET["train_labels"]
TEST_L = DATASET["test_labels"]

print("training data shape:", TRAIN_D.shape, "rock samples:", TRAIN_L.sum())
print("test data shape:", TEST_D.shape, "rock samples:", TEST_L.sum())

def run_training(num_steps=2):
    with tf.Graph().as_default() as g:
        data = tf.placeholder(tf.float32, shape=((None,)+TRAIN_D.shape[1:]))
        lbl = tf.placeholder(tf.int64, shape=None)
        droprate = tf.placeholder(tf.float32)
        logits, l2reg = model_conv2d(data,droprate)# model_conv1d(data, droprate)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lbl,logits=logits))
        loss += LAMBDA*l2reg
        global_step = tf.Variable(0, name='global_step', trainable=False) # to track global step
        optimizer = tf.train.AdamOptimizer(ALPHA).minimize(loss, global_step=global_step)
        # testing:
        correct_predictions = tf.cast(tf.equal(tf.argmax(logits, 1), lbl), tf.float64)
        accuracy = tf.reduce_mean(correct_predictions)
        init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    with tf.Session(graph=g, config=config) as sess:
        sess.run(init)
        print("Initialized")
        for step in xrange(num_steps):
            offset = (step * BATCH_SIZE) % (TRAIN_L.shape[0] - BATCH_SIZE)
            batch_data = TRAIN_D[offset:(offset+BATCH_SIZE)]
            batch_labels = TRAIN_L[offset:(offset+BATCH_SIZE)]
            _, l, ac = sess.run([optimizer,loss,accuracy],
                                feed_dict={data:batch_data,lbl:batch_labels,droprate:(DROPOUT_RATE+1.0/(step+1))})
            #print("step ", step, " (logits ", offset, "-", offset+BATCH_SIZE,"):", "\t\tbatch accuracy: ", ac, "\tbatch loss:",l, sep="")
            if (step%100==0):
                print("step ", step, " (logits ", offset, "-", offset+BATCH_SIZE,"):", "\t\tbatch accuracy: ", ac, "\tbatch loss:",l, sep="")
                print("\n\tTEST ACCURACY:", sess.run(accuracy,
                                                   feed_dict={data:TEST_D,lbl:TEST_L,droprate:1.0}))
        print("FINAL TEST ACCURACY:", sess.run(accuracy,
                                               feed_dict={data:TEST_D, lbl:TEST_L, droprate:1.0}))

run_training(50000)





# # BATCH_SIZE = 20
# # ALPHA= 1e-4
# # LAMBDA=0.2
# # DROPOUT_RATE=1
# # DS_PATH = DATASET_STFTS_FILEPATH # DATASET_WAVCHUNKS_FILEPATH #
# def model_conv2d(batch, droprate_placeholder, hidden_size=24):
#     # conv layer1
#     W = weight_variable([512, 5, 1, 4])
#     b = bias_variable(4)
#     conv = relu(conv2d(batch, W, strides=[1,1,1,1], padding="VALID")+b)
#     #conv = max_pool(conv, ksize([1,4,1,1], strides=[1,4,1,1], padding="SAME"))
#     l2reg = l2loss(W)
#     # conv layer2
#     W = weight_variable([1, 10, 4,6])
#     b = bias_variable(6)
#     conv = relu(conv2d(conv, W, strides=[1,1,1,1], padding="VALID")+b)
#     l2reg += l2loss(W)
#     # conv layer3
#     W = weight_variable([1, 15, 6, 8])
#     b = bias_variable(8)
#     conv = relu(conv2d(conv, W, strides=[1,1,1,1], padding="VALID")+b)
#     l2reg += l2loss(W)
#     # conv layer4
#     W = weight_variable([1, 20, 8, 12])
#     b = bias_variable(12)
#     conv = relu(conv2d(conv, W, strides=[1,1,1,1], padding="VALID")+b)
#     l2reg += l2loss(W)
#     # output of CNN:
#     shape = conv.get_shape().as_list()
#     flatsize = shape[1]*shape[2]*shape[3]
#     conv_flat = tf.reshape(conv, [tf.shape(conv)[0], flatsize])
#     # fully connected hidden layer
#     W = weight_variable([flatsize, hidden_size])
#     b = bias_variable(hidden_size)
#     hidden = relu(matmul(conv_flat, W)+b)
#     l2reg += l2loss(W)
#     # dropout&logits
#     W = weight_variable([hidden_size, 2])
#     b = bias_variable(2)
#     l2reg += l2loss(W)
#     return tf.matmul(dropout(hidden, droprate_placeholder), W)+b, l2reg
