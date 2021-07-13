#!/usr/bin/bash

echo "Getting embeddings..."
# word2vec
wget -nc "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
zcat GoogleNews-vectors-negative300.bin.gz > word2vec.bin 

# emoji2vec
wget -nc  "https://github.com/uclnlp/emoji2vec/raw/master/pre-trained/emoji2vec.bin" 