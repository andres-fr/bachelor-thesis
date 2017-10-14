"""This file contains some functionality implemented in the context of the bachelor thesis, but
   not used in the final version. See the docstrings for the details.
"""


####################################################################################################
### 6) RAAGA CLASSIFICATION: SPLIT LABELS INTO TRAINING AND TEST SUBSETS
####################################################################################################

RAAGA_LABELS_PATH = DATASETS_PATH+"raaga_labels.pickle"
TRAIN_CV_TEST = [0.7, 0.1, 0.2] # ratios to split the dataset

def calculate_and_save_raaga_labels(outpath):
    """Do this only once: project sanitized labels as a dict of key=raaga, value=list of
      (id, dur_in_secs) tuples sorted by duration. After that, it can be reloaded with
       raaga_labels = load_pickle(RAAGA_LABELS_PATH)
    """
    labels = load_pickle(CLEAN_LABELS_PICKLE)
    wavs = load_labeled_wavs(labels)
    raaga_labels = filter_labels(labels, wavs, "raaga") # class, id, seconds
    # dictionary holding all the distinct raagas: fill it with (id, seconds) labels
    raagas = {r[0]:[] for r in raaga_labels}
    for l in raaga_labels:
        raagas[l[0]].append(l[1:])
    # for each raaga, sort the list by duration and export to outpath
    raagas = {k:sorted(v, key=lambda x:x[1]) for k,v in raagas.iteritems()}
    save_pickle(raagas,  outpath)


def split_labels_into_train_and_test(all_labels, train_ratio=TRAIN_CV_TEST[0],
                                     out_path=None, verbose=False):
    """Inputs: a dictionary of the form class:(label_id, seconds) and 0 < train_ratio < 1.
       As it can be seen in the histograms, the different ragas aren't uniformly represented in the
       dataset. The chosen strategy to overcome this problem is here to perform data augmentation.
       But augmented data cannot appear in the test set, so both subsets have to be separated before
       augmentating, or cutting the recordings into chunks.

       Furthermore, the durations of the recordings within each raaga are also imbalanced, which
       could mean, for instance, that one single recording represents the 30% of the class, or
       conversely, that most of the augmentation of the trainset is based on a single recording.
       This would skew the generalization ability of the model and the result measurements.
       To achieve a healthy balance between quantity and variety of the data in both training and
       test subsets, before data augmentation, the following rule is followed for every raaga:

          1) loop over the raaga's samples sorted from shortest to longest duration
          2) every N+(r%N) samples, take one into the test set (see further explanation)
          3) stop when the test ratio for that raaga is surpassed
          4) if any of both subsets remains empty, print a warning for the corresponding raaga

       This approach ensures a healthy balance between quantity and variety of the data in both
       training and test subsets. The N+(r%N) formula ensures two things: firstly, N is inversely
       proportional to the test_ratio, so the duration distribution is as uniform as possible in
       both subsets. Secondly, (r%N) is a random integer between 0 and N-1, that prevents the
       subset distribution to be always the same (which could affect the generalization ability to
       other datasets).

       This function returns two dictionaries of key=raaga_uuid, value=list of (id,dur_in_secs) labels.
       For each raaga, the sum of the dur_in_secs of the first dictionary should be, approximately,
       equal to the train_ratio parameter (the durations for the second dict would be then
       1-train_ratio).

       Usage: split_labels_into_train_and_test(load_pickle(RAAGA_LABELS_PATH), TRAIN_CV_TEST[0])
    """
    if(train_ratio<=0 or train_ratio>=1):
        raise RuntimeError("~split_labels_into_train_and_test: train_ratio must be in (0,1) and was "
                           +str(train_ratio))
    # main parameters for the algorithm
    test_ratio = 1-train_ratio
    test_freq = int(1.0 / test_ratio)
    rand = np.random.randint(test_freq) # integer in [0, test_freq]
    total_durations = {k:sum([x[1] for x in v]) for k,v in all_labels.iteritems()}

    # dict construction!
    train_labels = {l:[] for l in all_labels}
    test_labels = {l:[] for l in all_labels}
    for k,v in all_labels.iteritems(): # iter over every raaga
        for i, label in enumerate(v): # iter over every raaga's labels
            # if it is the test's turn, and the raaga's test subset isn't full yet, add to test
            if( (((i+rand)%test_freq)==0) and
                sum([x[1] for x in test_labels[k]])<(total_durations[k]*test_ratio) ):
                test_labels[k].append(label)
            # else add to train
            else:
                train_labels[k].append(label)
    # check if any class has no representation at all, and warn
    train_without_labels = {k for k,v in train_labels.iteritems() if not v}
    test_without_labels = {k for k,v in test_labels.iteritems() if not v}
    if train_without_labels:
        print("\n~WARNING in split_labels_into_train_and_test:", len(train_without_labels),
              "classes have no TRAINING data:", train_without_labels,
              "\nto avoid this try choosing a train_ratio closer to 0.5")
    if test_without_labels:
        print("\n~WARNING in split_labels_into_train_and_test:", len(test_without_labels),
              "classes have no TEST data:", test_without_labels,
              "\nto avoid this try choosing a test_ratio closer to 0.5")
    # for the classes with representation in both subsets, print/export some stats:
    represented = set(all_labels).difference(train_without_labels).difference(test_without_labels)
    pretty_table = []
    for k in represented:
        trainNr =  len([x[1] for x in train_labels[k]])
        traindur =  sum([x[1] for x in train_labels[k]])
        testNr =  len([x[1] for x in test_labels[k]])
        testdur =  sum([x[1] for x in test_labels[k]])
        realdur = traindur+testdur
        expected_dur = total_durations[k]
        pretty_table.append([k, realdur, trainNr, traindur, testNr, testdur,
                             traindur/float(realdur), expected_dur-realdur<0.1])
    pretty_table = tabulate(sorted(pretty_table, key=lambda x:x[5], reverse=True),
                            headers=["raaga", "totaldur", "trainNr.","traindur","testNr.","testdur",
                                     "train_ratio","nothing_lost?"])
    pretty_table = ("Dataset after calling split_labels_into_train_and_test, with train_ratio ="+
                    str(train_ratio)+"\n"+pretty_table)
    if verbose: print(pretty_table)
    if out_path:
        with open(out_path+".txt", "w") as f:
            f.write(pretty_table.encode("UTF-8"))
        NotImplemented # seaborn functionality! http://seaborn.pydata.org/examples/heatmap_annotation.html
        # import seaborn as sns
        # sns.set()
        # pass
    return train_labels, test_labels
