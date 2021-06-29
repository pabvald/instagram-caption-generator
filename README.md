# instagram-caption-generator


## Links 

- Overleaf reports:
  1. Proposal: https://es.overleaf.com/8613455172kzxdmwbkzwff
  2. Interim report: https://es.overleaf.com/6554477761qnybhyqpqxyz
  3. Final report (with specified template): https://www.overleaf.com/6613286691kbtfvgcbhwcd

- Presentations - Google Drive: https://drive.google.com/drive/folders/1kR7Vl8Tv27xR9-1w7cdbKSqXYVx05k4p?usp=sharing

- Docker Pytorch images -  DockerHub: https://hub.docker.com/r/pabvald/pytorch/tags?page=1&ordering=last_updated


## Preprocessing 

The `preprocessing/` folder contains two Python scripts that allow to preprocess the 
[flickr8k](https://www.kaggle.com/adityajn105/flickr8k) , [flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset) and the [instagram](https://www.kaggle.com/prithvijaunjale/instagram-images-with-captions) datasets, assuming that they are stored in the folders  `data/datasets/flickr8k`, `data/datasets/flickr30k` and `data/datasets/instagram`, respectively,  as follows:

```bash
python preprocessing/flickr.py -d flickr8k # preprocess 'flickr8k' dataset

python preprocessing/flickr.py -d flickr30k # preprocess 'flickr30k' dataset

python preprocessing/instagram.py  # preprocess 'instagram' dataset
```


The parameters that can be specified are:

- `-min` or `--minimal-length`: minimum length of the captions. The default is `2`.
- `-max` or `--maximal-length`: maximum length of the captions. The default is `50`.
- `-wf` or `--min-word-frequency`: minimum frequency of a word to be included in the word map / vocabulary.
- `-c` or `--captions-per-image`: number of captions per image. The default is `5`.

Besides, when running the `preprocessing/instagram.py` script, the following additional parameters can be specified:
- `-t` or `--train-size`: proportion of the dataset that is used for training. The default is `0.60` (60%).
- `-v` or `--val-size`: proportion of the dataset that is used for training. The default is `0.20` (20%). The size of the test split will be computed as `1 - (train-size + val-size)`.

and when running the `preprocessing/flickr.py` script, the additional parameter can be specified: 
- `-d` or `--dataset` with possible values `'flickr8k'` or `'flickr30k'`. Default is `'flickr8k'`.

The output of this process includes the following files:

- word map (`WORDMAP_datasetname.json`): a .json file containing a mapping word: index.
- preprocessed images (`SPLIT_IMAGES_datasetname.hdf5`) for the `TRAIN`, `VAL` and `TEST` splits.
- encoded captions (`SPLIT_CAPLENS_datasetname.json`) for the `TRAIN`, `VAL` and `TEST` splits.  These files contain the encoded captions with a fix length equal to the `--maximal-length` argument. The captions are encoded using the word map in `WORDMAP_dataset.json`. 
