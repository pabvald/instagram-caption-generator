#!/usr/bin/env python3

import os
import torch
import json
from config import *
from os.path import join as pjoin
from utils import plot_history



#=================
# Configuration
#=================
os.environ['TORCH_HOME'] = 'pretrained' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

#=================
# Data parameters
#=================
data_folder = PATH_FLICKR  # folder with data files saved by create_input_files.py
data_name = 'flickr8k'  # base name shared by data files


#=================
# Model parameters
#=================
emb_dim = 300  # dimension of word embeddings
attention_dim = 300  # dimension of attention linear layers
decoder_dim = 300  # dimension of decoder RNN
dropout = 0.5

#====================
# Training parameters
#====================
start_epoch = 0
epochs = 2  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
patience = 20  # early stopping patience
batch_size = 80
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
fine_tune_embeddings = False  # fine-tine embeddings?
checkpoint = None  # path to checkpoint, None if none
train_name = "bs{}_ad{}_dd{}_elr{}_dlr{}".format(batch_size, attention_dim, decoder_dim, encoder_lr * int(fine_tune_encoder), decoder_lr)

save_dir = os.path.join(PATH_MODELS, data_name, train_name)


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


with open(pjoin(PATH_MODELS, data_name, 'BEST_checkpoint_bs80_ad300_dd300_elr0.0_dlr0.0004.pth.tar')) as infile:
    checkpoint = json.load(infile)

#--------------------------------
# Create the plots
#--------------------------------

# plot the loss history
history = checkpoint['history']
plot_history(save_dir, history)