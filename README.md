# P-SIF: Document Embeddings using Partition Averaging


## Introduction
  - For text classification and information retrieval tasks, text data has to be represented as a fixed dimension vector. 
  - We propose simple feature construction technique named **P-SIF: Document Embeddings using Partition Averaging**
  - We demonstrate our method through experiments on multi-class classification on 20newsGroup dataset, multi-label text classification on Reuters-21578 dataset, Semantic Textual Similarity Tasks (STS 12-16) and other classification tasks.

## Testing
There are 3 folders named 20newsGroup, Reuters and STS which contains code related to multi-class classification on 20newsGroup dataset, multi-label classification on Reuters dataset, and Semantic Texual Similarity Task (STS) on 27 datasets.

Create word weights
```sh
$create_wrd_cnt.py
```

Get word vectors for all words in vocabulary: 
```sh
$ python Word2Vec.py
```

Get Sparse Document Vectors (SCDV) for documents in train and test set:
```sh
$ python ksvd_sif.py
```

Get performance metrics on test set:
```sh
$ python models.py
```

#### Other_Datasets
For running P-SIF on rest of the 7 datasets, go to Other_Datasets folder. 
Inside Other_Datasets folder, each dataset has a folders with the dataset name. 
Follow the Readme.md has been included for running the P-SIF. 
You have to download google embedding from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing and placed in the Other_Dataset folder.

## Requirements
Minimum requirements:
  -  Python 3.6+
  -  NumPy 1.8+
  -  Scikit-learn
  -  Pandas
  -  Gensim
