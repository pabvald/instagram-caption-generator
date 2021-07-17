#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import os
import sys
import csv
import torch
import argparse
import numpy as np
import pandas as pd
#import pytesseract as tess

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from config import *
from collections import Counter
from gensim.models import KeyedVectors
from preprocessing import create_input_files
from utils import create_wordmap, load_embeddings

#tess.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

# Remove warnings
pd.options.mode.chained_assignment = None  # default='warn'

# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--minimal-length", default=2)
parser.add_argument("-max", "--maximal-length", default=50)
parser.add_argument("-wf", "--min-word-frequency", default=5)
parser.add_argument("-c", "--captions-per-image", default=1)
parser.add_argument("-t", "--train-size", default=0.65)
parser.add_argument("-v", "--val-size", default=0.15)
args = vars(parser.parse_args())

# Paths
DIR = os.path.dirname(__file__)
PATH_SLANG = pjoin(DIR, '../data/preprocessing/slang.txt')
PATH_SYNONYMS = pjoin(DIR, '../data/preprocessing/synonyms_en.txt')

# Constants
RAND_STATE = 42
PUNCTUATIONS = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
MIN_WORD_FREQ = None #int(args['min_word_frequency'])
CAPTIONS_PER_IMAGE = int(args['captions_per_image'])
CAPT_MIN_LENGTH = int(args['minimal_length']) 
CAPT_MAX_LENGTH = int(args['maximal_length'])
TRAIN_SIZE = float(args['train_size'])
VAL_SIZE = float(args['val_size'])


# Functions
def slang_translator(filepath, text):
    """ Translate slang to normal text.
        :param filepath: path to the .txt files containing the slang translations
        :param text: text to translate
    """
   
    with open(filepath, "r") as csv_file:
        # Reading file as CSV with delimiter as "=", 
        # so that abbreviation are stored in row[0] and phrases in row[1]
        data_from_file = csv.reader(csv_file, delimiter="=")

        words = text.split()        
        for j, word in enumerate(text.split()):         
            # Removing Special Characters.
            word = re.sub('[^a-zA-Z0-9-_.]', '', word)
            for row in data_from_file:
                # Check if selected word matches short forms[LHS] in text file.
                f_word = word.strip().upper()
                if f_word == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    words[j] = row[1] 
        
    return(' '.join(words))

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

            if actual_length < CAPT_MIN_LENGTH or actual_length > CAPT_MAX_LENGTH:
                data.drop(row, axis=0, inplace=True)
            else:           
                #lengths.append(actual_length)
                data['Caption'][row] = new_caption

    data = data.reset_index()       
    data.drop(['index'], axis=1, inplace=True) 
    data.dropna(inplace=True)

    #data['Length'] = lengths

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
    """ Split dataset in train, validation and test set
        :param data
    """
    train, val, test = np.split(data.sample(frac=1, random_state=RAND_STATE),
                                    [int(TRAIN_SIZE*len(data)), int((TRAIN_SIZE + VAL_SIZE)*len(data))])

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

def augment_caption(data, synfile, caption_number = 1):
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

def preprocess(data):
    """ Preprocess captions and remove images that doesn't meet the requierements
        :param data: 
    """
    print("Preprocessing dataset....")
    
    # load word2vec and emoji2vec embeddings
    print("\tLoading embeddings...")
    wv = KeyedVectors.load_word2vec_format(PATH_WORD2VEC, binary=True)
    ev = KeyedVectors.load_word2vec_format(PATH_EMOJI2VEC, binary=True)

    # load .csv file containing image locations and captions 
    print("\tLoading captions...")
    data = pd.read_csv(pjoin(PATH_INSTA, 'captions_csv.csv'))

    # preprocess the images
    # print("\tPreprocessing images...")
    # data = preprocess_images(data)

    # preprocess the captions
    print("\tPrepocessing captions...")
    data = preprocess_captions(data, wv, ev)
     
    return data 

def main():
    """ MAIN """

    # load .csv, preprocess images and captions if necessary
    print("Loading Instagram dataset...")
    if  os.path.isfile(pjoin(PATH_INSTA, "preprocessed.csv")):
        data = pd.read_csv(pjoin(PATH_INSTA, 'preprocessed.csv'))
    else:
        data = pd.read_csv(pjoin(PATH_INSTA, 'captions_csv.csv'))
        data = preprocess(data)
        data.to_csv(pjoin(PATH_INSTA, "preprocessed.csv"))
   
    # split the dataset in train, val, test
    print("Splitting the dataset in train, trainval, val and test ...")
    splits = split_dataset(data)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()
    synonyms = _get_synonyms(PATH_SYNONYMS)

    for split, dataset in splits.items():        
        for _, row in dataset.iterrows():
            captions = [] 
            path = pjoin(PATH_INSTA, row[1] + '.jpg')
            for i in range(CAPTIONS_PER_IMAGE):                
                caption = row[2]
                if i > 0: 
                    caption = _augment_sentence(synonyms, len(caption), caption)
                
                tokens = caption.split()
                word_freq.update(tokens)
                if len(tokens) <= CAPT_MAX_LENGTH:
                    captions.append(tokens)

            if len(captions) == 0:
                continue

            if split in {'train'}:
                train_image_paths.append(path)
                train_image_captions.append(captions)
            elif split in {'val'}:
                val_image_paths.append(path)
                val_image_captions.append(captions)
            elif split in {'test'}:
                test_image_paths.append(path)
                test_image_captions.append(captions)
        
    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map, save it to json
    word_map = create_wordmap("instagram", word_freq, PATH_INSTA, MIN_WORD_FREQ)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    for impaths, imcaps, split in [ (train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:
        
        create_input_files("instagram", impaths, imcaps, split, word_map, 
                            PATH_INSTA, CAPTIONS_PER_IMAGE, CAPT_MAX_LENGTH)

    # Create embeddings
    _, embeddings, emb_dim = load_embeddings(word_map=word_map)
    torch.save(embeddings, pjoin(PATH_INSTA, 'EMBEDDINGS_instagram.pt'))

if __name__ == '__main__':
    main()