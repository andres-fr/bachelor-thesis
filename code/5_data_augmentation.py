"""This file contains some functionality implemented in the context of the bachelor thesis, but
   not used in the final version. See the docstrings for the details.
"""


####################################################################################################
### AUGMENTATION OF TRAINING SUBSET
####################################################################################################

RUBBERBAND_BIN = "../rubberband/bin/rubberband"
AUG_DS_PATH = DATASETS_PATH+"augmented_carnatic_wavs_for_gulati_raagas.h5"

def stretch_and_transpose_wav(pathin, samplerate, speed_factor=1, semitones_shift=0, label="tmpstrch"):
    """Given the path to a wav file and the speed/semitone changes, returns a 1D numpy array of
       length (len(original_wav)/speed_factor), with the transposed frequencies (12=octave up,
       -12= octave down, etc). More specifically:
         1) tries to call rubberband with the given parameters and throws an exception if failure
         2) if success, generates the altered file in the /tmp directory using the <LABEL>+timestamp
         3) samples down the altered file to samplerate using ffmpeg (has to be installed)
         4) loads the transformed file and returns it as numpy 1D float array
         5) if everything went well, removes all generated files in tmp
       For that, it calls the rubberband binary, a GNU-licensed program that can be found in GitHub:
                           https://github.com/jlank/rubberband 
       Rubberband allows independent speed and pitch alterations with very good results. This can
       be used in data augmentation to balance the classes and reinforce the data consistency, since
       various of the real world audible categories are invariant to those two factors.
       Usage example: twice as fast, but pitch 2 tones lower
         stretch_and_transpose_wav(DATASETS_PATH+'a_song.wav', 2, -4)
    """
    tmpname = lambda: "/tmp/"+label+"_"+str(time.time())+"_"+str(random.randint(0,100000000))+".wav"
    ### 1. call rubberband to generate altered file to tmp dir
    name1 = tmpname()
    rubberband_command = (RUBBERBAND_BIN+" -T "+str(speed_factor)+" -p "+str(semitones_shift)+
                         " "+str(pathin)+" "+name1)
    if(system(rubberband_command) != 0):
        raise RuntimeError("~stretch_and_transpose_wav: ERROR calling "+ rubberband_command)
    ### 2. if successful, call ffmpeg to sampledown file from 44100 to samplerate
    name2 = tmpname()
    sampledown_command = ("ffmpeg -i "+name1+" -ac 1 -ar "+str(samplerate)+" "+name2  +
                          " < /dev/null > /dev/null 2>&1")
    if(system(sampledown_command) != 0):
        raise RuntimeError("~stretch_and_transpose_wav: ERROR calling "+ sampledown_command)
    # if successful so far, load file to numpy array
    arr = load_wav_file(name2, samplerate)
    ### 3. delete .wav files generated in tmp
    remove_command = "rm "+name1+" "+name2
    if(system(remove_command) != 0):
        raise RuntimeError("~stretch_and_transpose_wav: ERROR calling "+ rubberband_command)
    # return array
    return arr

def uniform_augmentation(pathin, speed_exp_limit=0.3, semitones_lin_limit=7):
    """Given a path to a wav file, and the stretching&transposition limits as positive floats, calls
       stretch_and_transpose_wav to modify the wav file based on values drawn from a uniform
       distribution X ~ unif(-1, 1), more specifically:
          speed ~ 2**(speed_exp_limit*X).
          transposition ~ semitones_lin_limit*X
       This means: if both values are 0, no modification will be applied. If
       semitones_lin_limit=12, the transposition will be between 1 octave up and 1 octave down.
       If speed_exp_limit=1, the stretching would be between twice and half as long.
       Returns a tuple with the new file as numpy 1D float array, the stretch factor
       as second element, and the transposition factor as third one.
    
       Some forms of augmentation may skew the data, by introducing redundancies or unfeasible
       transformations. In the case of carnatic raaga classification, time-stretching may
       be reasonable, especially due to the fact that raagas are more persistent to mild tempo
       changes than, for instance, taalas (although not independent: Gulati, p.41: "temporal
       aspects of melody are fundamental in characterization of raagas). Transposition could be
       feasible for other kinds of data, but could be also problematic for some applications
       because the vocal formants, some string instruments and such may not be invariant to it.
       Depending on the model architecture, linear transposition may even have a similar effect
       to a plain duplication.
       Without knowing the tempo management of carnatic music in detail, the default values are
       believed to provide a fair balance between augmentation variety and feasibility of the
       results (this has to be cross-validated anyway). Usage example:
         truncnorm_stretch_augmentation(CARNATIC_WAV_PATH+some_id+'.wav')
    """
    print("called augmentationFn for", pathin)
    # ensure that limits are understood correctly by the unif distribution
    speed_exp_limit = abs(speed_exp_limit)
    semitones_lin_limit = abs(semitones_lin_limit)
    unif = lambda mul: np.random.uniform(-mul, mul)
    # calculate the speed_factor
    stretch_factor = 2.0**unif(speed_exp_limit)
    transp_factor = unif(semitones_lin_limit)
    return (stretch_and_transpose_wav(pathin, SAMPLERATE, stretch_factor, transp_factor),
            stretch_factor, transp_factor)


def augmentate_class_of_trainset(trainset_labels, class_uuid):
    """Given:
         * trainset _labels: a dictionary in the form  key=raaga_uuid, val=list of (id,seconds).
           Note that the one generated by split_labels_into_train_and_test has raaga_name and NOT
           raga_uuid as key, so it has to be transformed before.
         * a keyword of the trainset_labels dictionary indicating which class is to be augmented
       This function performs data augmentation on the wav files of the training subset for the
       indicated class, and returns a 'CLASS_UUID||WAV_ID' : wav_array dictionary holding all the
       original+augmented samples that correspond to the given dictionary and class.
       All the wavs listed in trainset_labels are loaded using the load_labeled_wavs function,
       which takes a list of the IDs.
       The augmentation is done after the following assumptions and criteria:
         1) Assuming that the total duration amongst classes is imbalanced, and that data
            augmentation without profound knowledge of the data domain is risky and should be
            minimized, the amount of augmentation here should be the least necessary to achieve
            balance among the inter-class total durations, and should introduce only feasible
            real-world transformations and as less redundancy as possible. See the docstring of
            hte stretch_augmentation_function for more on this.
         2) Intra-class duration imbalance is also assumed to be bad: augmentation should also have
            the goal of compensating this. But, conversely, naive augmentation proportional to the
            length can be also bad: very short recordings would be repeated too many times, which
            would neglect the effect of the transformation, and end up being too redundant. This
            should also be avoided.
         3) Augmentation may/should be performed at the preprocessing stage, before cutting the
            songs into chunks and calculating other representation forms (like the STFT or melgram):
              a) The stretch_and_transpose_wav function is too slow to be performed for each batch
                 at learning stage. Also, it only works for .wav files.
              b) Stretching the data alters its length, and chunks are expected to have the same
                 length, so stretching after chunking is problematic.
              c) Storing the augmented data is impractical for big datasets, but feasible here.
    
       This function, similar to the one used to separate training and test datasets, achieves a
       good compromise for the given criteria, under the given assumptions: the amount of
       augmentation is close to the minimal needed, achieves an inter-class balance close to ideal,
       and promots inter-class balance while tending to avoid high redundancies. More precisely:
         1. get the <MAX_DUR> of the most represented class
         2. for each raaga in train_labels:
            2.1 calculate the <AVG_DUR> of each song so that the class achieves <MAX_DUR>
            2.2 sort the recordings by ascending duration
            2.3 while <RAAGA_DUR> < <MAX_DUR>, cyclic loop over the raaga's recordings:
                add (WAVID_STRETCH_TRANSP : AUGMENTED) to dictionary and increment <RAAGA_DUR>
    
        Usage example:
          tl, _ = split_labels_into_train_and_test(load_pickle(RAAGA_LABELS_PATH), TRAIN_CV_TEST[0])
          augmentate_class_of_trainset(tl, u'Bhairavi')
    """
    SR_FLOAT = float(SAMPLERATE)
    # first load the wavs referred by the given labels: key=wav_id, val=wav_array
    # this is also the returned dict
    result = load_labeled_wavs([x[0] for x in trainset_labels[class_uuid]])
    # augmentation depends on the total duration of the most represented class. calculate it:
    class_durs = {k:sum([x[1] for x in v]) for k,v in trainset_labels.iteritems()}
    max_dur = float(max(class_durs.values())) # for Todi raaga, 115*0.7 around 80k seconds
    # stats for the selected class
    class_dur = class_durs[class_uuid]
    avg_dur = max_dur/len(result)
    # cyclic loop over the sorted recs in ascending order: augment recs until max_dur is surpassed
    sorted_reclabels = sorted(trainset_labels[class_uuid], key=lambda x: x[1], reverse=False)# list of (label_id, seconds)
    cycle = len(sorted_reclabels)
    i=0
    while (class_dur<max_dur):
        print("\n\n",class_uuid, "ENTERED WHILE:  current_aug_dur", class_dur, "  goal:", max_dur)
        # variables for the currently observed recording
        rec_id = sorted_reclabels[i][0]
        rec_dur = sorted_reclabels[i][1]
        # the actual data augmentation part stretch_factor, transp_factor)
        arr, stretch, transp = uniform_augmentation(CARNATIC_WAV_PATH+rec_id+'.wav')
        result[rec_id+"_"+str(stretch)+"_"+str(transp)] = arr
        class_dur += arr.size/SR_FLOAT
        # loop further
        i = (i+1)%cycle
    # FINALLY: transform the keys of the result dict (key=id, value=wav_array) into the string
    # "CLASS_UUID||WAV_ID", to be able to save it to h5 format
    for k in result.keys(): result[class_uuid+"||"+k] = result.pop(k)
    return result



def augmentate_and_export_gulati(outpath, num_threads=20):
    """Given an absolute path:
          1) Calculates the train dataset resulting of split_labels_into_train_and_test
          2) Calls augmentate_class_of_trainset for the raagas present in Gulati's PhD(parallelized)
          3) Exports result to outpath as h5 format.
       The result is a dict in the form key='raaga_name||wav_id', val=wav_array. This function
       takes long to evaluate and is intended to be performed just once. Usage example:
       AUG_DS_PATH = DATASETS_PATH+"augmented_carnatic_wavs_for_gulati_raagas.h5"
       augmentate_and_export_gulati(AUG_DS_PATH)
       After that, the augmentated dataset can be reloaded with:
       aug_ds =load_h5_dict(AUG_DS_PATH)
    """
    all_labels_by_raaga = load_pickle(RAAGA_LABELS_PATH)
    train_labels_by_raaga,_ = split_labels_into_train_and_test(all_labels_by_raaga,TRAIN_CV_TEST[0])
    train_gulati_labels = {l["uuid"] : train_labels_by_raaga[l["name"]] for l in RAAGAS_GULATI}
    AUG_DATASET = {}
    print(train_gulati_labels.keys())
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for d in executor.map(lambda x:augmentate_class_of_trainset(train_gulati_labels, x),
                              train_gulati_labels.keys()):
            AUG_DATASET.update(d)
        save_h5_dict(AUG_DATASET, outpath)

def test_augmentation_in_one_class(class_uuid='85ccf631-4cdf-4f6c-a841-0edfcf4255d1',
                                   h5_path="/tmp/test_augm.h5"):
    """loads all labels, splits test labels, augmentates one single class, exports to h5, loads it
       again and export some of its samples to wav, that can be tested by hearing. Usage example:
         test_augmentation_in_one_class()
    """
    tl, _ = split_labels_into_train_and_test(load_pickle(RAAGA_LABELS_PATH), TRAIN_CV_TEST[0])
    d =  augmentate_class_of_trainset(tl, selected_class)
    save_h5_dict(d, h5_path)
    aug_ds =load_h5_dict(h5_path)
    for k in aug_ds.keys()[0:50]:
        raaga, wavid = k.split("||")
        pywav.write(wavid+".wav", 8000, d[k])


# # do this only once (per augmentation):
# # after done, recall augmented dataset ("CLASS||WAVID" : wav_arr) with:
# # aug_ds =load_h5_dict(AUG_DS_PATH)
# augmentate_and_export_gulati(AUG_DS_PATH, 60)

