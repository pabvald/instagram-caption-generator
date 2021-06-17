#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import json
import re
import pandas as pd
import argparse
from PIL import Image
from gensim.models import KeyedVectors
from os.path import join as pjoin
#import pytesseract as tess
#tess.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

# remove warnings
pd.options.mode.chained_assignment = None  # default='warn'

# parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-ml", "--minimal-length", default=2)
args = vars(parser.parse_args())

# paths
PATH_CAPTIONS = '../data/datasets/instagram/captions'
PATH_SLANG= '../data/preprocessing/slang.txt'
PATH_WORD2VEC = '../data/embeddings/word2vec.bin'
PATH_EMOJI2VEC = '../data/embeddings/emoji2vec.bin'

# constants
PUNCTUATIONS = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
CAPT_MIN_LENGTH = int(args['minimal_length']) 


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

    for row, caption in enumerate(data['Caption']):
        if type(caption)!=str or len(caption.split()) < CAPT_MIN_LENGTH:
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
                data['Caption'][row] = new_caption

    data = data.reset_index()       
    data.drop(['index'], axis=1, inplace=True) 

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

def main():
    """ MAIN """
    print("Preprocessing Instagram dataset....")
    
    # load word2vec and emoji2vec embeddings
    print("\tLoading embeddings...")
    wv = KeyedVectors.load_word2vec_format(PATH_WORD2VEC, binary=True)
    ev = KeyedVectors.load_word2vec_format(PATH_EMOJI2VEC, binary=True)

    # load .csv file containing image locations and captions 
    print("\tLoading captions...")
    data = pd.read_csv(pjoin(PATH_CAPTIONS, 'captions_csv.csv'))

    # preprocess the captions
    print("\tPrepocessing captions...")
    data = preprocess_captions(data, wv, ev)

    # preprocess the images
    # print("\tPreprocessing images...")
    # data = preprocess_images(data)

    # save preprocessed captions to .csv
    print("\tSaving preprocessed dataset...")
    data.to_csv(pjoin(PATH_CAPTIONS, "preprocessed_captions.csv"))

    # save preprocessed captions to .json with evaluation format
    data_dict = dict()
    for index, row in data.iterrows():
        data_dict[row[0]] = [row[2]]

    with open(pjoin(PATH_CAPTIONS, "references.json"), 'w') as fp:
        json.dump(data_dict, fp, sort_keys=True, indent=4)


# run main 
main()

