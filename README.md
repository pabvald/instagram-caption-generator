# Deep Insta: Instagram Caption Generator

## Contents

[***Jupyter Notebooks***](#jupyter-notebooks)

[***Preprocessing***](#preprocessing)

[***Training***](#training)

[***Evaluation***](#evaluation)

[***Acknowledgements - Honor Code***](#honor-code)


## Jupyter Notebooks

The `notebooks/` folder contains the following Jupyter Notebooks:

- [ev_metrics_summary.ipynb](./notebooks/ev_metrics_summary.ipynb): includes an explanation of the four evaluation metrics that we have used (BLEU, METEOR, CIDEr and ROUGE-L) with references to the corresponding *papers* as well as examples of how to compute these metrics using Python libraries or the `evaluation` model.

- [instagram_captions.ipynb](./notebooks/instagram_captions.ipynb): a few statistics regarding the captions of the `instagram` dataset.

- [visualize_results.ipynb](./notebooks/visualize_results.ipynb): shows how to use the code in [caption.py](./caption.py) to visualize the *attention* process of the model when predicting its caption.

- [word_embeddings.ipynb](./notebooks/word_embeddings.ipynb): little demostration of how to load and use the word embeddings models of `word2vec` and `emoji2vec`.


## Preprocessing 

The `preprocessing/` folder contains three Python scripts that allow to preprocess the 
[flickr8k](https://www.kaggle.com/adityajn105/flickr8k) , [flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset) and the [instagram](https://www.kaggle.com/prithvijaunjale/instagram-images-with-captions) datasets, assuming that they are stored in the folders  `data/datasets/flickr8k`, `data/datasets/flickr30k` and `data/datasets/instagram`, respectively,  as follows:

```bash
python preprocessing/flickr.py -d flickr8k # preprocess 'flickr8k' dataset

python preprocessing/flickr.py -d flickr30k # preprocess 'flickr30k' dataset

python preprocessing/instagram.py  # preprocess 'instagram' dataset

python preprocessing/flickr-insta.py #preprocess and combine the 'flickr8k' (or 'flickr30k') and the 'instagram dataset
```

The parameters that can be specified are:

- `-min` or `--minimal-length`: minimum length of the captions. The default is `2`.
- `-max` or `--maximal-length`: maximum length of the captions. The default is `50`.
- `-wf` or `--min-word-frequency`: minimum frequency of a word to be included in the word map / vocabulary.
- `-c` or `--captions-per-image`: number of captions per image. The default is `5`. However, be aware that the `instagram` dataset contains only one caption per image.

Besides, when running the `preprocessing/instagram.py` script, the following additional parameters can be specified:
- `-t` or `--train-size`: proportion of the dataset that is used for training. The default is `0.60` (60%).
- `-v` or `--val-size`: proportion of the dataset that is used for training. The default is `0.20` (20%). The size of the test split will be computed as `1 - (train-size + val-size)`.

and when running the `preprocessing/flickr.py` script, the additional parameter can be specified: 
- `-d` or `--dataset` with possible values `'flickr8k'` or `'flickr30k'`. Default is `'flickr8k'`.

The output of this process includes the following files:

- word map (`WORDMAP_datasetname.json`): a .json file containing a mapping word - index.
- preprocessed images (`SPLIT_IMAGES_datasetname.hdf5`) for the `TRAIN`, `VAL` and `TEST` splits.
- encoded captions (`SPLIT_CAPLENS_datasetname.json`) for the `TRAIN`, `VAL` and `TEST` splits.  These files contain the encoded captions with a fix length equal to the `--maximal-length` argument. The captions are encoded using the word map in `WORDMAP_dataset.json`. 


## Training

The [train.py](./train.py) script allows to train a model from scratch or continue training a model providing a certain *checkpoint*. Due to the high number of parameters, these can be directly modified in the code.  Check the code to see which parameters can be specified.

Once the parameters have been fixed, you can execute the following command to train the model: 
```python 
python train.py 
```

Note: the code assumes that a file `EMBEDDINGS_dataset.pt` containing the embeddings exists and loads it. However, is it possible to train its own embeddings from scratch.

## Evaluation 
The [evaluate.py](./evaluate.py) scripts to evaluate a model providing the correspoding checkpoint and making use of the `evaluation` module. By default, all the metrics are calculated: BLEU (1, 2, 3 and 4), METEOR, CIDEr and ROUGE-L. Check the code to see which parameters can be specified

Once the parameters have been fixed, you can execute the following command to train the model: 
```python 
python evaluate.py 
```

## Honor Code 

This repository includes code from the following repositories: 

- Microsoft COCO Caption Evaluation, from Tsung-Yi Lin: https://github.com/tylin/coco-caption
- image-caption-metrics repository, from EricWWWW: https://github.com/EricWWWW/image-caption-metrics
- A PyTorch tutorial for Image Captioning, from Sagar Vinodababu: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning