import h5py
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from os.path import join as pjoin
from random import seed, choice, sample
from collections import Counter


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


def encode_caption(caption, word_map, capt_max_length):
    """Encode a caption given a word mapping 
        :param caption: list of tokens
        :param word_map: word mapping
        :param capt_max_length: maximum length of a caption
    """
    return [word_map['<start>']] + \
           [word_map.get(word, word_map['<unk>']) for word in caption] + \
           [word_map['<end>']] + [word_map['<pad>']] * (capt_max_length - len(caption))


def create_input_files(dataset, impaths, imcaps, split, word_map, output_folder, captions_per_image, capt_max_length):
    """ Create input files.
        :param dataset: the name of the dataset
        :param impaths: images' paths 
        :param imcaps: images' captions 
        :param split: train, val or test
        :param word_map
        :param output_folder
        :param captions_per_image
        :param capt_max_length 
    """
    print("Creataing files for {} dataset".format(dataset))
    seed(123)
    with h5py.File(pjoin(output_folder, split + '_IMAGES_' + dataset + '.hdf5'), 'a') as h:
        # Make a note of the number of captions we are sampling per image
        h.attrs['captions_per_image'] = captions_per_image

        # Create dataset inside HDF5 file to store images
        images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

        print("\nReading {} images and captions, storing to file...\n".format(split))

        enc_captions = []
        caplens = []

        for i, path in enumerate(tqdm(impaths)):

            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)

            # Sanity check
            assert len(captions) == captions_per_image

            # Read images
            img = np.asarray(Image.open(impaths[i]))
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            img = np.resize(img, (256, 256, 3))
            img = img.transpose(2, 0, 1)
            assert img.shape == (3, 256, 256)
            assert np.max(img) <= 255

            # Save image to HDF5 file
            images[i] = img

            for j, c in enumerate(captions):
                # Encode captions
                enc_c = encode_caption(c, word_map, capt_max_length)

                # Find caption lengths
                c_len = len(c) + 2

                enc_captions.append(enc_c)
                caplens.append(c_len)

        # Sanity check
        assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

        # Save encoded captions and their lengths to JSON files
        with open(pjoin(output_folder, split + '_CAPTIONS_' + dataset + '.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(pjoin(output_folder, split + '_CAPLENS_' + dataset + '.json'), 'w') as j:
            json.dump(caplens, j)
