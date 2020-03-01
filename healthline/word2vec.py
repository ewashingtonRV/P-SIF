#!/usr/bin/env python
import pandas as pd
import logging
from gensim.models import Word2Vec
import time
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import sys
import csv

def custom_tokenize(text):
    if not text:
        print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return word_tokenize(text)


if __name__ == '__main__':
    start = time.time()
	# The csv file might contain very huge fields, therefore set the field_size_limit to maximum.
    csv.field_size_limit(sys.maxsize)
    novant = pd.read_csv("/Users/ewashington/Desktop/clean_novant.csv")
    tokens = novant['text'].apply(custom_tokenize)
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
    
    # Params
    num_features = 100  # Word vector dimensionality
    min_word_count = 5  # Minimum word count
    num_workers = 2  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    epochs = 20
    
    print("Training Word2Vec model...")
	# Train Word2Vec model.
    model = Word2Vec(tokens, workers=num_workers, hs=0, sg=1, negative=10,
                     iter=epochs, size=num_features, min_count=min_word_count,
                     window=context, sample= downsampling, seed=1313)
	           
	                
    # Save Word2Vec model.
    print("Saving Word2Vec model...")
    model_name = "./saved_models/word2vec"
    model.init_sims(replace=True)
    model.save(model_name)
    endmodeltime = time.time()

    print("time : ", endmodeltime-start)
