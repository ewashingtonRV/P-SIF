# P-SIF: Document Embeddings using Partition Averaging


## Introduction
  - This model was highlighted at AAAI 2020: https://vgupta123.github.io/docs/AAAI-GuptaV.3656.pdf
  - For text classification and information retrieval tasks, text data has to be represented as a fixed dimension vector. 
  - Simple feature construction technique named **P-SIF: Document Embeddings using Partition Averaging**

## Notes
Still a work in progress. Path to data needs to be entered in `Word2Vec.py` and `psif.py` and several lines need to be uncommented. Files are currently set to load previously intialized and saved models. 

## Run file
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
