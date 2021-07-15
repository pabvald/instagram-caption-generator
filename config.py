import os
from os.path import join as pjoin


# paths
DIR = os.path.dirname(__file__)
PATH_MODELS = pjoin(DIR, 'models')
PATH_VECS = pjoin(DIR, 'data/embeddings')
PATH_WORD2VEC = pjoin(DIR, 'data/embeddings/word2vec.bin')
PATH_EMOJI2VEC = pjoin(DIR, 'data/embeddings/emoji2vec.bin')
PATH_DATASETS = pjoin(DIR, 'data/datasets')
PATH_INSTA = pjoin(DIR, 'data/datasets/instagram')
# PATH_FLICKR = pjoin(DIR, 'data/datasets/flickr8k/min_freq=5')
PATH_FLICKR = pjoin(DIR, 'data/datasets/flickr8k/95vocab')
PATH_FLICKR_INSTA = pjoin(DIR, 'data/datasets/flickr8k-insta')

PATH_FLICKR30 = pjoin(DIR, 'data/datasets/flickr30k')
EVAL_IMAGES_PATH = pjoin(DIR, 'eval_images')