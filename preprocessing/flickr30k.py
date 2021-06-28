#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import argparse
from os.path import join as pjoin
from collections import Counter
from common import create_wordmap, create_input_files

# parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-min", "--minimal-length", default=2)
parser.add_argument("-max", "--maximal-length", default=50)
parser.add_argument("-wf", "--min-word-frequency", default=5)
parser.add_argument("-c", "--captions-per-image", default=1)
args = vars(parser.parse_args())


# paths 
PATH_KAPATHY = '../data/datasets/flickr30k/dataset_flickr30k.json'
IMG_FOLDER = '../data/datasets/flickr30k/img/flickr30k_images'
OUTPUT_FOLDER = '../data/datasets/flickr30k'

# constants
RAND_STATE = 42
CAPTIONS_PER_IMAGE = int(args['captions_per_image'])
MIN_WORD_FREQ = int(args['min_word_frequency'])
CAPT_MIN_LENGTH = int(args['minimal_length']) 
CAPT_MAX_LENGTH = int(args['maximal_length'])


def main():
    """ MAIN """

    # Read Karpathy JSON
    with open(PATH_KAPATHY, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for caption in img['sentences']:
            # Update word frequency
            word_freq.update(caption['tokens'])
            if len(caption['tokens']) <= CAPT_MAX_LENGTH:
                captions.append(caption['tokens'])

        if len(captions) == 0:
            continue

        path = pjoin(IMG_FOLDER, img['filename'])

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
    word_map = create_wordmap("flickr30k", word_freq, MIN_WORD_FREQ, OUTPUT_FOLDER)
  
    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    for impaths, imcaps, split in [ (train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        create_input_files("flickr30k", impaths, imcaps, split, word_map, 
                            OUTPUT_FOLDER, CAPTIONS_PER_IMAGE, CAPT_MAX_LENGTH)

if __name__ == '__main__':
    main()