#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Author: Sagar Vinodababu (@sgrvinod)
    Source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
"""

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import *
from config import *
from tqdm import tqdm
from evaluation.eval import eval as evaluate_metrics
from dataloader import CaptionDataset
from os.path import join as pjoin
from nltk.translate.bleu_score import corpus_bleu
from torchvision.utils import save_image



# Parameters
metrics = ['bleu', 'cider', 'rouge', 'meteor'] # select the desired metrics
data_folder = PATH_FLICKR  # folder with data files saved by create_input_files.py
data_name = 'flickr8k'  # base name shared by data files
checkpoint = pjoin(PATH_MODELS, data_name, 'BEST_checkpoint_bs80_ad300_dd300_elr0.0_dlr0.0004.pth.tar')  # model checkpoint
word_map_file = pjoin(data_folder, 'WORDMAP_' + data_name + '.json')  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
inv_word_map = inverse_word_map(word_map)
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size, metrics, verbose=False):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    test_data = CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = dict()
    hypotheses = dict()

    # Determine captions per image
    image, caps, caplens, allcaps =  next(iter(test_loader))
    caps_per_img = allcaps[0].shape[0]

    # For each image
    #for i, (image, caps, caplens, allcaps) in enumerate([next(iter(test_loader))]):
    for index, (image, caps, caplens, allcaps) in enumerate(tqdm(test_loader)):

        if index % caps_per_img != 0: 
            continue

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds].long()]
            c = c[prev_word_inds[incomplete_inds].long()]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        try:
            j = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[j]
        except ValueError:
            continue

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(map(lambda c: decode_caption(c, word_map, inv_word_map), img_caps))
        references[str(index)] = img_captions
        
        # Hypotheses
        hypotheses[str(index)] = [decode_caption(seq, word_map, inv_word_map)]   
        assert len(references) == len(hypotheses)

        # Print results
        if verbose and (index % 50 == 0):
            print("\n References:")
            for r in img_captions:
                print(" - ", r)
            print("Hypothesis:\n - ", hypotheses[str(index)])

            # save_image(image, pjoin(EVAL_IMAGES_PATH, '{}.png'.format(i)))
            # print('\nImage: {}'.format(i))
            # print('The real sentence:      {}'.format(decode_caption(caps[0], word_map, inv_word_map)))
            # scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(torch.tensor([image]).to(device), caps.to(device), caplens.to(device))
            # print('The generated sentence: {}\n'.format(caps_sorted[0]))
    
    # Calculate metrics
    bleu4 = corpus_bleu(list(references.values()), list(hypotheses.values()))
    results = evaluate_metrics(references, hypotheses, metrics=metrics)   
    results['nltk-bleu'] = bleu4

    return results

if __name__ == '__main__':
    beam_size = 1
    metrics = evaluate(beam_size, metrics, verbose=True)
    
    print("\n Evaluation metrics with beam size = {}".format(beam_size))
    for k, v in metrics.items():
        print(" - {} = {}".format(k.upper(), v))

   
    