import h5py
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import encode_caption
from os.path import join as pjoin
from random import seed, choice, sample


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
    print("Creating files for {} dataset".format(dataset))
    seed(123)
    with h5py.File(pjoin(output_folder, split + '_IMAGES_' + dataset + '.hdf5'), 'a') as h:
        # Make a note of the number of captions we are sampling per image
        h.attrs['captions_per_image'] = captions_per_image

        # Create dataset inside HDF5 file to store images
        images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

        print("\nReading {} images and captions, storing to file...\n".format(split))

        enc_captions = []
        caplens = []

        for i in tqdm(range(len(impaths))):

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
