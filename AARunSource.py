######################################################################################
# Floris Fok
# Final bacherlor project
#
# 2019 febuari
# Transfer learning from medical and non medical data sets to medical target data
#
# ENJOY
######################################################################################
# Logic of pretraining and convential training
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


def run_source(params):
    """ 
    Run reference experiment or make a pre-trained model, All the logic takes place here
    Input: params(dict)
    Output: Model saved in json and h5, txt and figure.
    """
    
    ## Allow expension of VRAM ##
    # fixes allowcation problems
    config_desktop()

    # A data generator could also work, but is much slower. 
    # If Vram or Ram of your computer is not large enough, please do use a data generator.

    ## Dowload Data & split data ##
    x,y = get_data(params)
    x_test,y_test,x,y = val_split(x,y, params["test_size"])
    x_val,y_val,x,y = val_split(x,y, params["val_size"])
   
    ## Make VGG16 model, with three additional dense layers to fit class count ##
    model = make_model(x, y, params)

    ## Weights are determined bye differences in classes in the training data, currently with float accuracy ##
    weights = determen_weights(y)
    H, score, model = train_model(model,x,y,x_val,y_val,x_test,y_test, params, weights)
    
    ## Save model to JSON ##
    model_json = model.to_json()
    with open(f"{params['model_path']}{params['epochs']}_{params['Data']}.json", "w") as json_file:
        json_file.write(model_json)

    ## Serialize weights to HDF5 ##
    model.save_weights(f"{params['model_path']}{params['epochs']}_{params['Data']}_Weights.h5")
    print("Saved model to disk")

    ## Document results and params used ##
    results = {'score':score,"acc_epoch":H.history['acc'],"val_acc_epoch":H.history['val_acc'],"loss_epoch":H.history['loss'],"vall_loss_epoch":H.history['val_loss']}
    doc(params,results,H,params["doc_path"])