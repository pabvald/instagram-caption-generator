#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from config import *
from collections import Counter
from preprocessing.instagram import preprocess, split_dataset
from preprocessing import create_input_files
from utils import create_wordmap, load_embeddings


# Remove warnings
pd.options.mode.chained_assignment = None  # default='warn'

# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--minimal-length", default=2)
parser.add_argument("-max", "--maximal-length", default=50)
parser.add_argument("-wf", "--min-word-frequency", default=5)
parser.add_argument("-c", "--captions-per-image", default=3)
parser.add_argument("-t", "--train-size", default=0.65)
parser.add_argument("-v", "--val-size", default=0.15)
args = vars(parser.parse_args())

# Paths
DIR = os.path.dirname(__file__)
PATH_SLANG = pjoin(DIR, '../data/preprocessing/slang.txt')
PATH_SYNONYMS = pjoin(DIR, '../data/preprocessing/synonyms_en.txt')
PATH_KAPATHY_SPLIT = pjoin(PATH_FLICKR, 'dataset_flickr8k.json')
FLICKR_IMG_FOLDER = pjoin(PATH_FLICKR, 'img')

# Constants
DATA_NAME = 'flickr8k-insta'
RAND_STATE = 42
MIN_WORD_FREQ =  None #int(args['min_word_frequency'])
CAPTIONS_PER_IMAGE = int(args['captions_per_image'])
CAPT_MIN_LENGTH = int(args['minimal_length']) 
CAPT_MAX_LENGTH = int(args['maximal_length'])
TRAIN_SIZE = float(args['train_size'])
VAL_SIZE = float(args['val_size'])


def main():
    """ MAIN """

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    #========================== INSTAGRAM ================================

    # load .csv, preprocess images and captions if necessary
    print("Loading Instagram dataset...")
    if  os.path.isfile(pjoin(PATH_INSTA, "preprocessed.csv")):
        data = pd.read_csv(pjoin(PATH_INSTA, 'preprocessed.csv'))
    else:
        data = pd.read_csv(pjoin(PATH_INSTA, 'captions_csv.csv'))
        data = preprocess(data)
        data.to_csv(pjoin(PATH_INSTA, "preprocessed.csv"))
   
    # split the dataset in train, val, test
    splits = split_dataset(data)  

    for split, dataset in splits.items():        
        for _, row in dataset.iterrows():
            captions = [] 
            path = pjoin(PATH_INSTA, row[1] + '.jpg')              
            caption = row[2]
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
    
    # ======================= FLICKR8K =============================
    print("Loading Flickr dataset...")
    # Read Karpathy JSON
    with open(PATH_KAPATHY_SPLIT, 'r') as j:
        data = json.load(j)

    for img in data['images']:
        captions = []
        for caption in img['sentences']:
            # Update word frequency
            word_freq.update(caption['tokens'])
            if len(caption['tokens']) <= CAPT_MAX_LENGTH:
                captions.append(caption['tokens'])

        if len(captions) == 0:
            continue

        path = pjoin(FLICKR_IMG_FOLDER, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map, save it to json
    word_map = create_wordmap(DATA_NAME, word_freq, PATH_FLICKR_INSTA, MIN_WORD_FREQ)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    for impaths, imcaps, split in [ (train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:
        
        create_input_files(DATA_NAME, impaths, imcaps, split, word_map, 
                            PATH_FLICKR_INSTA, CAPTIONS_PER_IMAGE, CAPT_MAX_LENGTH)

    # Create embeddings
    _, embeddings, emb_dim = load_embeddings(word_map=word_map)
    torch.save(embeddings, pjoin(PATH_FLICKR_INSTA, 'EMBEDDINGS_' + DATA_NAME + '.pt'))

if __name__ == '__main__':
    main()