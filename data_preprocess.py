#!/home/ykhassanov/.conda/envs/py37/bin/python
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import librosa
import pdb, json
import numpy as np
from hparam import hparam as hp
from utils import extract_all_feat

def save_spectrogram_tisv_train():
    """ Full preprocess of text independent utterance.
        The spectogram or log-mel-spectrogram is saved as numpy file.
    """
    print("start text independent utterance feature extraction for the train set")
    audio_path = glob.glob(hp.data.train_path_unprocessed + '/*')
    # make folder to save processed train features
    os.makedirs(os.path.join(hp.data.train_path, hp.data.feat_type), exist_ok=True)

    train_speaker_num = len(audio_path)
    print("*train speaker number: %d"%train_speaker_num)
    for i, folder in enumerate(audio_path):
        print("%dth speaker processing..."%i)
        os.makedirs(os.path.join(hp.data.train_path, hp.data.feat_type, "speaker%d"%i), \
                    exist_ok=True)
        for j, utter_name in enumerate(glob.glob(folder+'/*')):
            if utter_name[-5:] == '.flac':

                S = extract_all_feat(utter_name, mode = 'train')
                np.save(os.path.join(hp.data.train_path, hp.data.feat_type, "speaker%d"%i, \
                                     "utt%d.npy"%j), S)


def save_spectrogram_tisv_test():
    """ Full preprocess of text independent utterance.
        The spectogram or log-mel-spectrogram is saved as numpy file.
    """
    print("start text independent utterance feature extraction for the test set")
    # make folder to save processed test file
    os.makedirs(os.path.join(hp.data.test_path, hp.data.feat_type), exist_ok=True)    

    verify_list = np.loadtxt(hp.data.test_meta_path, str)
    labels = np.array([int(i[0]) for i in verify_list])
    list1 = np.array([os.path.join(hp.data.test_path_unprocessed, i[1]) for i in verify_list])
    list2 = np.array([os.path.join(hp.data.test_path_unprocessed, i[2]) for i in verify_list])

    #triplets = []
    #c = 0
    for i, (l, p1, p2) in enumerate(zip(labels, list1, list2)):
        s1 = extract_all_feat(p1, mode = 'test').transpose()    #dim: time, spec
        s2 = extract_all_feat(p2, mode = 'test').transpose()
        triplet = np.array([l, s1, s2])
        np.save(os.path.join(hp.data.test_path, hp.data.feat_type, "test_triplet%d.npy"%i), triplet)
        #triplets.append([l, s1, s2])
        #c += 1
        #if c>=100:
        #    break

    #triplets = np.array(triplets)
    #print(triplets.shape)
    #np.save(os.path.join(hp.data.test_path, hp.data.feat_type, "test_triplets.npy"), triplets)

def save_spectrogram_tisv_test_Libri():
    print("start test independent utterance feature extraction for the train set")
    audio_path = glob.glob(hp.data.test_path_unprocessed + '/*')
    # make folder to save processed train features
    os.makedirs(os.path.join(hp.data.test_path, hp.data.feat_type), exist_ok=True)

    total_num = len(audio_path)
    print("*test total number: %d"%total_num)

    for i, name in enumerate(audio_path):
        print("%d speaker processing..."%i)
        if name[-5:] == '.flac':
            S = extract_all_feat(name, mode = 'train')
            np.save(os.path.join(hp.data.test_path,"utt%d.npy"%i), S)


if __name__ == "__main__":
    # save_spectrogram_tisv_train()
    save_spectrogram_tisv_test_Libri()

