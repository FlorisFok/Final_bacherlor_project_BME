######################################################################################
# Floris Fok
# Final bacherlor project
#
# 2019 febuari
# Transfer learning from medical and non medical data sets to medical target data
#
# ENJOY
######################################################################################
# logic of experiments using tranfer learning
######################################################################################
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import tensorflow as tf
from AADatasets import *
from AAPreTrain import *
from AATransferLearn import *
from AAlogic import *
from AALabnotesDoc import *

def run_target(params):
    """ 
    Run transfer learning experiment with a certain paramerter set. All the logic takes place here
    Input: params (dict)
    Output: Save Figure and text file with results
    """

    ## Dowload Data & split data ##
    x,y = get_data(params)
    x_test,y_test,x,y = val_split(x,y, params["test_size"])
    x_val,y_val,x,y = val_split(x,y, params["val_size"])
    
    ## Allow expension of VRAM ##
    # fixes allowcation problems
    config_desktop()

    # A data generator could also work, but is much slower. 
    # If Vram or Ram of your computer is not large enough, please do use a data generator.


    ## Dowload pre-trained model ##
    model = make_model(x, y, params)

    if params["style"] == 'FT':
        ## For experiments WITH fine-tuning ##
        ## Weights are determined bye differences in classes in the training data, currently with float accuracy ##
        weights = determen_weights(y)

        ## Re-train the model ##
        H, score, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params, weights)

        ## Extract the Feature vector ##
        ## Layer can be changed ##
        predictions = get_feature_vector(model, x, layer = 'fc2')
        predictions_test = get_feature_vector(model, x_test, layer = 'fc2')

        ## Train SVM and Get AUC ##
        score = auc_svm(predictions,y,predictions_test,y_test, plot = False)
        results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss']}

    elif params["style"] =='SVM':
        ## For experiments WITHOUT fine-tuning ##
        ## Extract the Feature vector ##
        ## Layer can be changed ##
        predictions = get_feature_vector(model, x, layer = 'fc2')
        predictions_test = get_feature_vector(model, x_test, layer = 'fc2')

        ## Train SVM and Get AUC ##
        score = auc_svm(predictions,y,predictions_test,y_test, plot = False)
        results = {'score':score,'data_name':params["data_name"],'method':params["model"],'style':params["style"]}
        H=None

    ##Save results and params to a txt ##
    doc(params,results,H,params["doc_path"])