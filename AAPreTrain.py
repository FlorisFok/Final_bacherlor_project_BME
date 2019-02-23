######################################################################################
# Floris Fok
# Final bacherlor project
#
# 2019 febuari
# Transfer learning from medical and non medical data sets to medical target data
#
# ENJOY
######################################################################################
# Functions concering pretraining and training
######################################################################################    
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import sklearn
from sklearn import datasets, svm, metrics
from sklearn.metrics import roc_auc_score
from sklearn import metrics


def make_model(x, y,params):
    '''
    Makes model from scratch or makes model from pre-trained one, delivers it ready to train. IF you want to change the optimizer, you should change it here.
    Input: image data, class data, Params
    Output: Trainable model
    '''

    ## For imagenet we use the keras pre-trained model, so it has its own section ##
    if params['model'] == 'imagenet':
        classes = 1000
        vgg_conv = tf.keras.applications.VGG16(weights=params["model"],input_shape = (x[0].shape), include_top=True, classes=classes)
        fine_tune = tf.keras.layers.Dense(500, activation='relu')(vgg_conv.output)
        fine_tune = tf.keras.layers.Dense(350, activation='relu')(fine_tune)
        fine_tune = tf.keras.layers.Dense(y.shape[1], activation='sigmoid')(fine_tune)
        vgg_conv  = tf.keras.models.Model(inputs=vgg_conv.input, outputs=fine_tune)

    ## For training from scratch with VGG16 ##
    elif params['model'] == "None":
        c = int(y.shape[1])
        vgg_conv = tf.keras.applications.VGG16(weights=None,input_shape = (x[0].shape), include_top=True, classes=c)        

    ## For loading pre-trained models ##
    else:
        with open(params['model_path'][params['model']], 'r') as json_file:
            loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        #Load weights into new model
        loaded_model.load_weights(f"{params['model_path'][params['model']][:-5]}_Weights.h5")
        fine_tune = tf.keras.layers.Dense(y.shape[1], activation='sigmoid')(loaded_model.output)
        vgg_conv  = tf.keras.models.Model(inputs=loaded_model.input, outputs=fine_tune)

    ## IF you want to know if the model is created correctly ##
    # print("MODEL SUMMARY:")
    # for layer in vgg_conv.layers:
    #     print(layer, layer.trainable)
    # print("END OF SUMMARY")

    ## Compile it with its optimizer and loss function ##
    opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.01, decay=0, nesterov=True)
    # Cat_cross = best for AUC, there is no AUC optimalisation
    vgg_conv.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return vgg_conv

def train_model(model,x_train,y_train,x_val,y_val,x_test,y_test, params,weights_dict = None):
    """
    Trains or re-trains model, contains early stopping and the possibility to use Best model only.
    Input:
    Output:
    """
    ## Extract some params to make it more readable ##
    Epochs = params["epochs"]
    Batch_size = params["Batch_size"]

    ## Callbacks! These can be switch on or off, early stop is recommended, save best is optional ##
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
    check = tf.keras.callbacks.ModelCheckpoint(r"C:\models\model_temp.h5", monitor='val_acc', verbose=1, save_best_only=True)
    ## Trains the desired callbacks ##
    if params['stop'] == 'no':
        H = model.fit(x_train, y_train, batch_size=Batch_size, epochs=Epochs, validation_data=(x_val,y_val),shuffle=True,  callbacks = [], class_weight = weights_dict)
    if params['stop'] == 'best':
        H = model.fit(x_train, y_train, batch_size=Batch_size, epochs=Epochs, validation_data=(x_val,y_val),shuffle=True,  callbacks = [stop, check], class_weight = weights_dict)
        ## To use best model ##
        model.load_weights(r"C:\models\model_temp.h5")
    else:
        H = model.fit(x_train, y_train, batch_size=Batch_size, epochs=Epochs, validation_data=(x_val,y_val),shuffle=True,  callbacks = [stop], class_weight = weights_dict)
    
    ## Calculate the Area Under the Curve
    score = roc_auc_score(y_test, model.predict(x_test))
    return H, score, model

def data_generator_large_files(pathes, batch_size):
    '''
    This is a data generator. Use this if your computer has memory issues.
    '''

    while True: #generators for keras must be infinite
        for path in pathes:
            x_train, y_train = prepare_data(path)

            totalSamps = x_train.shape[0]
            batches = totalSamps // batch_size

            if totalSamps % batch_size > 0:
                batches+=1

            for batch in range(batches):
                section = slice(batch*batch_size,(batch+1)*batch_size)
                yield (x_train[section], y_train[section])

def config_desktop():
    ## WHEN USING TF 1.5 or lower and GPU ###
    #                                       #
    config = tf.ConfigProto()               #               
    config.gpu_options.allow_growth = True  #
    session = tf.Session(config=config)     #
    #########################################