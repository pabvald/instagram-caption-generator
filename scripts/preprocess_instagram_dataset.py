#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import json
import re
import numpy as np
import pandas as pd
import argparse
from gensim.models import KeyedVectors
from os.path import join as pjoin
#import pytesseract as tess
#tess.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

# remove warnings
pd.options.mode.chained_assignment = None  # default='warn'

# parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-ml", "--minimal-length", default=2)
parser.add_argument("-t", "--train-size", default=0.6)
parser.add_argument("-v", "--val-size", default=0.2)
args = vars(parser.parse_args())

# paths
PATH_CAPTIONS = '../data/datasets/instagram'
PATH_SLANG= '../data/preprocessing/slang.txt'
PATH_WORD2VEC = '../data/embeddings/word2vec.bin'
PATH_EMOJI2VEC = '../data/embeddings/emoji2vec.bin'
PATH_SYNONYMS = '../data/preprocessing/synonyms_en.txt'

# constants
PUNCTUATIONS = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
CAPT_MIN_LENGTH = int(args['minimal_length']) 
RAND_STATE = 42
TRAIN_SIZE = float(args['train_size'])
VAL_SIZE = float(args['val_size'])
AUG_N = 2
# TEST_SIZE = 1 - TRAIN_SIZE - VAL_SIZE


# functions
def slang_translator(filepath, text):
    """ Translate slang to normal text.
    :param filepath: path to the .txt files containing the slang translations
    :param text: text to translate
    """
    text = text.split(" ")

    for j, _str in enumerate(text):       
        access_mode = "r"  # File Access mode [Read Mode]
        with open(filepath, access_mode) as csv_file:
            # Reading file as CSV with delimiter as "=", 
            # so that abbreviation are stored in row[0] and phrases in row[1]
            data_from_file = csv.reader(csv_file, delimiter="=")
            # Removing Special Characters.
            _str = re.sub('[^a-zA-Z0-9-_.]', '', _str)
            for row in data_from_file:
                # Check if selected word matches short forms[LHS] in text file.
                if _str.upper() == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    text[j] = row[1]

    return(' '.join(text))

def preprocess_captions(data, word2vec, emoji2vec):
    """ Preprocess the captions: remove non-english or punctuation characters,
    remove captions shorter than the minimum, lowercase captions """
    lengths = []

    for row, caption in enumerate(data['Caption']):
        if type(caption) != str or len(caption.split()) < CAPT_MIN_LENGTH:
            data.drop(row, axis=0, inplace=True)
        else:
            # slang translation
            caption = slang_translator(PATH_SLANG, caption)

            new_caption = ""
            for _char in caption:
                # ignore non-english and punctuation characters
                if (ord(_char)<127 and ord(_char)>31) and (_char not in PUNCTUATIONS):
                    new_caption += _char  
                # create tokens by considering  emojis as separate word
                elif ord(_char) > 120000 and ord(_char) < 130000:
                    new_caption += (" " + _char + " ")  
            
            # lowercasing
            new_caption = new_caption.lower()
            
            # check that a minimum of words have an embedding, discard if not
            actual_length = 0
            for word in new_caption.split():
                if (word in word2vec) or (word in emoji2vec):
                    actual_length += 1

            if actual_length < CAPT_MIN_LENGTH:
                data.drop(row, axis=0, inplace=True)
            else:           
                lengths.append(actual_length)
                data['Caption'][row] = new_caption

    data = data.reset_index()       
    data.drop(['index'], axis=1, inplace=True) 

    data['Length'] = lengths

    return data

# def preprocess_images(data):
#     """ Preprocess the images: remove the images that contain more than one word of text """
#     text_in_images = []
#     for i in data['Image File']:
#             img = Image.open(i +'.jpg')
#             text = tess.image_to_string(img)
#             text_in_images.append(text)
#     del_row = 0
#     for j in text_in_images:
#         x = j.strip().replace('\n'," ").replace('\x0c',"").strip()
#         if len(x.split()) > 2:
#             data.drop(del_row, axis=0, inplace=True)
#         del_row = del_row + 1

#     data = data.reset_index()       
#     data.drop(['index'], axis=1, inplace=True)

#     return data

def split_dataset(data):
    """Split dataset in train, validation and test set
    :param data
    """
    train, val, test = np.split(data.sample(frac=1, random_state=RAND_STATE),
                                    [int(TRAIN_SIZE*len(data)), int((TRAIN_SIZE+VAL_SIZE)*len(data))])

    splits = {'train': train,
             'val': val,
             'test': test,
             'trainval': pd.concat([train, val])
            }
    return splits

# Sample n random words and replace them on their synonyms
def _augment_sentence(sentence, n, synonyms):
    new_sent = sentence.copy()

    for i in range(n):
        if i > len(sentence):
            break

        idx = np.random.randint(len(new_sent))
        word = new_sent[idx]

        for syn_set in synonyms:
            if word in syn_set:
                new_word = np.random.choice([syn for syn in syn_set if syn != word])
                new_sent[idx] = new_word

    return new_sent

def _get_synonyms(path):
    with open(path) as f:
        syn = f.read().split('\n')
        synonyms = []
        for line in syn:
            syns = [w.strip() for w in line.split(',')]
            synonyms.append(syns)
    return synonyms

def augment_captions(data, synfile, caption_number = 1):
    """ Augment captions by synonyms replacement
        :param data
        :param synfile: Path to the file containing synonyms
        :param caption_number: Number of generated captions per image
        :param aug_n: Number of words to be replaced on their synonyms
    """
    synonyms = _get_synonyms(synfile)
    header = 'Caption_{}'.format(caption_number)
    augmented = {header: []}
    for caption in data['Caption']:
        if isinstance(caption, str):
            caption = caption.strip().split(' ')
        if not isinstance(caption, list):
            augmented[header].append(np.nan)
            continue

        new = _augment_sentence(caption, len(caption), synonyms)
        augmented[header].append(' '.join(new))

    aug_df = pd.DataFrame.from_dict(augmented)
    data = pd.concat([data, aug_df], axis=1)

    data.dropna(inplace=True)
    return data

def main():
    """ MAIN """
    print("Preprocessing Instagram dataset....")
    
    # load word2vec and emoji2vec embeddings
    print("\tLoading embeddings...")
    wv = KeyedVectors.load_word2vec_format(PATH_WORD2VEC, binary=True)
    ev = KeyedVectors.load_word2vec_format(PATH_EMOJI2VEC, binary=True)

    # load .csv file containing image locations and captions 
    print("\tLoading captions...")
    data = pd.read_csv(pjoin(PATH_CAPTIONS, 'preprocessed.csv'))

    # preprocess the captions
    print("\tPrepocessing captions...")
    data = preprocess_captions(data, wv, ev)

    # preprocess the images
    # print("\tPreprocessing images...")
    # data = preprocess_images(data)

    # Augment captions
    print("Augmenting the image captions...")
    data.dropna(inplace=True)
    for i in range(AUG_N):
        data = augment_captions(data, PATH_SYNONYMS, i)

    # split the dataset in train, val, test
    print("\tSplitting the dataset in train, val and test ...")
    splits = split_dataset(data)

    # save preprocessed captions to .csv
    print("\tSaving preprocessed dataset...")
    for split, dataset in splits.items():

        dataset.to_csv(pjoin(PATH_CAPTIONS, split + ".csv"))

        # save preprocessed captions to .json with evaluation format
        data_dict = dict()
        for index, row in dataset.iterrows():
            data_dict[row[0]] = [row[2]]

        with open(pjoin(PATH_CAPTIONS,  split + ".json"), 'w') as fp:
            json.dump(data_dict, fp, sort_keys=True, indent=4)


# run main 
main()