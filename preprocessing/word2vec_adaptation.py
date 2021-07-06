from gensim.models import KeyedVectors
from os.path import join as pjoin, dirname
import numpy as np

DIR = dirname(__file__)
PATH_WORD2VEC = pjoin(DIR, '../data/embeddings/word2vec.bin')

# load word2vec and emoji2vec embeddings
print("\tLoading embeddings...")
wv = KeyedVectors.load_word2vec_format(PATH_WORD2VEC, binary=True)
print(len(wv.vocab))
wv.add(['<start>', '<end>', '<pad>', '<unk>'], [np.random.dirichlet(np.ones(300), size=1)[0] for i in range(4)])
print(len(wv.vocab))
wv.save_word2vec_format(PATH_WORD2VEC[:-4]+'_updated.txt', binary=False)
