import json
from os.path import join as pjoin
from collections import Counter
import numpy as np
import torch
from gensim.models import KeyedVectors

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def save_checkpoint(save_dir, data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, pjoin(save_dir,'BEST_' + filename))

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def get_word_freqs(captions):
    """ Calculates word frequencies
        :param captions: list of captions
    """
    word_freqs = Counter()
    for caption in captions:
        word_freqs.update(list(filter(None, caption.split(' '))))
    return word_freqs


def create_wordmap(dataset, word_freq, output_folder, min_word_freq=None):
    """ Create a word map from a dictionary of the word frequencies and save it.
        :param dataset: name of the dataset
        :param word_freq: dictionary of word frequencies
        :param min_word_freq: minimum frequency of a word to be included in the map. If None, 95% of the vocabulary words will be included
        :param output_folder
    """

    if min_word_freq == None:  # Take 95% most common words from the vocabulary
        words_sorted = [word for word, freq in sorted(word_freq.items(), key=lambda item: item[1], reverse=True)]
        words = words_sorted[:int(0.95 * len(words_sorted))]  # The rest 5% set to '<unk>'
    else:
        words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]

    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    with open(pjoin(output_folder, 'WORDMAP_' + dataset + '.json'), 'w') as j:
        json.dump(word_map, j)

    return word_map

def _init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.
    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)

def load_embeddings(word_emb_file, emoji_emb_file, word_map=None, binary = True):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.
    :param word_emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map. If None, it will be comprised from the embeddings vocabulary
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    print("\tLoading embeddings...")
    wv = KeyedVectors.load_word2vec_format(word_emb_file, binary=binary)
    ev = KeyedVectors.load_word2vec_format(emoji_emb_file, binary=binary)
    # Find embedding dimension
    emb_dim = wv.vector_size

    if word_map == None:
        word_map = {k: v + 1 for emb in [wv.key_to_index.keys(), ev.key_to_index.keys()] for v, k in enumerate(emb)}
        word_map['<unk>'] = len(word_map) + 1
        word_map['<start>'] = len(word_map) + 1
        word_map['<end>'] = len(word_map) + 1
        word_map['<pad>'] = 0

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    _init_embedding(embeddings)

    # Iterate through the vector pairs
    for emb_word in vocab:

        if emb_word in wv.key_to_index:
            embeddings[word_map[emb_word]] = torch.FloatTensor(wv.get_vector(emb_word))
        elif emb_word in ev.key_to_index:
            embeddings[word_map[emb_word]] = torch.FloatTensor(ev.get_vector(emb_word))

    return word_map, embeddings, emb_dim #, wv, ev # wv and ev are for debugging

# DEBUGGING
# import os
# from os.path import join as pjoin
# import pandas as pd
#
# DIR = os.path.dirname(__file__)
# PATH_WORD2VEC = pjoin(DIR, '../data/embeddings/word2vec.bin')
# PATH_EMOJI2VEC = pjoin(DIR, '../data/embeddings/emoji2vec.bin')
#
# data = pd.read_csv(pjoin('../data/datasets/instagram/preprocessed.csv'))
#
# word_freqs = common.get_word_freqs(data['Caption'])
# wordmap = common.create_wordmap('instagram', word_freqs, '', min_word_freq=0)
# wordmap, embeddings, emb_dim = load_embeddings(PATH_WORD2VEC, PATH_EMOJI2VEC, word_map=None)
# print('Done')
# print(embeddings[:10])
# print(wordmap)
# print(len(embeddings))
# print(len(wordmap))
# print(wv.get_vector('cat'))
# print(embeddings[wordmap['cat']])
#
# print(ev.get_vector('💎'))
# print(embeddings[wordmap['💎']])

