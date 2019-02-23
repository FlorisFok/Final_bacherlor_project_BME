############################################################################################
# TEST IF ALL FUNCTIONS STILL WORK
#
#
############################################################################################

import os
import cv2
import sys
import time
import pickle
import pandas
import random
import sklearn

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import interp
from sklearn import metrics
from itertools import cycle
from sklearn import datasets, svm, metrics
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc,roc_auc_score

from AARunSource import *
from AARunTarget import *
from AADatasets import *
from AAPreTrain import *
from AALabnotesDoc import *
from AATransferLearn import *


def test():

    print("\n")

    params = {"Data":'CatDog',
            'file_path':r"C:\Users\s147057\Documents\BEP\pet set",
            'pickle_path':r"C:\Users\s147057\Documents\BEP",
            'model_path':r"C:\Users\s147057\Documents\BEP\models2_cat_dogmini.json",
            'doc_path':r"C:\Users\s147057\Documents\GitHub\t",
            'model':'imagenet',
            'img_size_x':224,
            'img_size_y':224,
            'norm':True,
            'color':True,
            'pretrain':None,
            "equal_data":False, 
            "shuffle": True, 
            "epochs": 2 , 
            "val_size":250,
            "test_size":300, 
            "Batch_size": 1,
            'RandomSeed':2,
            'stop':'no'
            }

    print("File path of data:")
    print(os.path.isdir(params["file_path"]))
    print("File path of pickle:")
    print(os.path.isdir(params["pickle_path"][:-len(params["Data"])]))
    print("File path of mdoel:")
    print(os.path.isdir(params["model_path"]))
    print("File path of Documentation:")
    print(os.path.isdir(params["doc_path"][:-1]))
    print("\n")
    print("Size of Image:")
    print(params['img_size_x'] > 223 and params['img_size_y'] > 223)
    print("\n")
    print("Testing functions:")

    x,y = get_data(params)
    if x == [] or y == []:
        print("get_data = not OK")
    else:
        print("get_data = OK")

    x_test, y_test, x, y = val_split(x, y, params['test_size'])
    x_val, y_val, x, y = val_split(x, y, params['val_size'])

    model = make_model(x, y, params)
    print("make_model = OK")

    weights = determen_weights(y)
    print("determen_weights = OK")

    H, score, model = train_model(model, x, y, x_val, y_val, x_test, y_test, params, weights_dict = None)
    print("train_model = OK")

    predictions = get_feature_vector(model, x, layer = 'fc2')
    predictions_test = get_feature_vector(model, x_test, layer = 'fc2')
    print("get_feature_vector = OK")

    score = auc_svm(predictions, y, predictions_test, y_test, plot = False)
    print("auc_svm = OK") 

    results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],
               "loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss'],
               'data_name':data_name,'method':method,'style':style}

    doc(params, results, H, params["doc_path"])
    print("doc = OK")

    print("ALL WAS OK")

if __name__ == '__main__':
    test()