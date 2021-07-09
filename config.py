from os.path import join as pjoin
import os

# paths
DIR = os.path.dirname(__file__)
PATH_WORD2VEC = pjoin(DIR, 'data/embeddings/word2vec.bin')
PATH_EMOJI2VEC = pjoin(DIR, 'data/embeddings/emoji2vec.bin')
PATH_INSTA = pjoin(DIR, 'data/datasets/instagram')
PATH_FLICKR = pjoin(DIR, 'data/datasets/flickr8k')