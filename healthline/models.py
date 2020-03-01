#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:14:29 2020

@author: ewashington
"""
import time
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import (GridSearchCV)
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.svm import LinearSVC


def label_encode(y_train, y_test):
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(y_train)
    encoded_y_train = label_encoder.transform(y_train)
    encoded_y_test = label_encoder.transform(y_test)
    return encoded_y_train, encoded_y_test
  
def get_max_predicted_k2(y_test_pred):
    k2_prediction = []
    for i in range(0, len(y_test_pred)):
        k2_prediction_i = np.argmax(y_test_pred[i])
        k2_prediction.append(k2_prediction_i)
    return np.array(k2_prediction) 

def calculate_k2_accuracy(y_test_pred, y_test):
    k2_prediction_array = get_max_predicted_k2(y_test_pred)
    k2_prediction_and_labels_df = pd.DataFrame(data=k2_prediction_array, columns=["k2_prediction"])
    k2_prediction_and_labels_df["label"] = y_test
    print("Accuracy: ", sum(k2_prediction_and_labels_df["k2_prediction"] == k2_prediction_and_labels_df["label"]) / len(k2_prediction_and_labels_df))

gwbowv = np.load("sdv_20cluster_100feature_matrix_ksvd_sparse.npy")
gwbowv_test = np.load("test_sdv_20cluster_100feature_matrix_ksvd_sparse.npy")

y_train = pd.read_csv("y_train.csv")
y_train = y_train.rename(columns={"Unnamed: 0": "index"})
y_train = y_train.drop(["index"], axis=1)

y_test = pd.read_csv("y_test.csv")
y_test = y_test.rename(columns={"Unnamed: 0": "index"})
y_test = y_test.drop(["index"], axis=1)

x_train = pd.DataFrame(gwbowv)
x_train = x_train.add_prefix('col_')
x_train["target"] = y_train

x_test = pd.DataFrame(gwbowv_test)
x_test = x_test.add_prefix('col_')
x_test["target"] = y_test

x_train.to_csv("gwbowv_train.csv")
x_test.to_csv("gwbowv_test.csv")

encoded_y_train, encoded_y_test = label_encode(y_train, y_test)
num_class = len(np.unique(encoded_y_train))

train_data = lgb.Dataset(gwbowv, label=encoded_y_train)
test_data = lgb.Dataset(gwbowv_test, label=encoded_y_test)

# lgb constants
n_rounds = 5000
early_stopping = 100

default_params = {
  'task': 'train',
  'boosting_type': 'gbdt',
  'objective': 'multiclass',
  'num_class': num_class,
  'metric': 'multi_logloss',
  'learning_rate': 0.05,
  'max_depth': 7,
  'num_leaves': 17,
  'feature_fraction': 0.4,
  'bagging_fraction': 0.6,
  'bagging_freq': 17}

default_model = lgb.train(default_params,
                       train_data,
                       valid_sets = test_data,
                       num_boost_round = n_rounds,
                       early_stopping_rounds = early_stopping)


default_y_test_pred = default_model.predict(gwbowv_test)
calculate_k2_accuracy(default_y_test_pred, encoded_y_test)

# SVC
param_grid = [
    {'C': np.arange(0.1, 7, 0.2)}]

scores = ['accuracy', 'recall_micro', 'f1_micro', 'precision_micro', 'recall_macro', 'f1_macro', 'precision_macro',
          'recall_weighted', 'f1_weighted', 'precision_weighted']  # , 'accuracy', 'recall', 'f1']

for score in scores:
    strt = time.time()
    print("# Tuning hyper-parameters for", score, "\n")
    clf = GridSearchCV(LinearSVC(C=1), param_grid, cv=5, scoring='%s' % score)
    # {'C': 3.900000000000001}
    clf.fit(gwbowv, y_train["tag"])
    print("Best parameters set found on development set:\n")
    print(clf.best_params_)
    print("Best value for ", score, ":\n")
    print(clf.best_score_)
    y_pred = clf.predict(gwbowv_test)
    print("Report")
    print(classification_report(y_test, y_pred, digits=6))
    print("Accuracy: ", clf.score(gwbowv_test, y_test))
    print("Time taken:", time.time() - strt, "\n")
endtime = time.time()
print("Total time taken: ", endtime - strt, "seconds.")

print("********************************************************")
