#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:07:16 2020

@author: ewashington
"""
#import gensim.downloader as api
#corpus = api.load('text8')
from gensim.models.word2vec import Word2Vec
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter


#nut = pd.read_csv("/Users/ewashington/Desktop/nutrition.csv")
novant = pd.read_csv("/Users/ewashington/Desktop/clean_novant.csv")


def custom_tokenize(text):
    if not text:
        print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return word_tokenize(text)


#corpus = novant["text"].tolist()
#df = novant['text'].dropna(inplace=True)
tokens = novant['text'].apply(custom_tokenize)
#c = sent_to_words(corpus)
# Preparing the dataset
#all_sentences = nltk.sent_tokenize(corpus)

token_lst = [item for sublist in tokens.to_list() for item in sublist]
cnt = Counter(token_lst).most_common()

with open("./data/wrd_cnt.txt","w") as f:
    for line in cnt:
        strs = " ".join(str(x) for x in line)
        f.write(strs+"\n")
f.close()

