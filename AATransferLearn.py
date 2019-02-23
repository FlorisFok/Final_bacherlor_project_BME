######################################################################################
# Floris Fok
# Final bacherlor project
#
# 2019 febuari
# Transfer learning from medical and non medical data sets to medical target data
#
# ENJOY
######################################################################################
# Functions concering transferlearning
######################################################################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from itertools import cycle
import sklearn
from sklearn import datasets, svm, metrics
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

def get_feature_vector(model, x, layer):
    '''
    choose certain layer and extract output of given vector
    input: model, image data, layer
    ouput: feature vector
    '''
    #resize model
    model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer).output)
    #get vector
    predictions = model.predict(x)
    return predictions

def preform_svm(x,y,x_val,y_val):
    # OLD SVM NO AUC (prototype)
    # preform the svm on the vector
    '''
    Simple support vector machine, Variable zijn nog aanpasbaar, maar nog niet mee geexperimenteerd. pakt automatisch een deel test en train. (afb moet vierkant zijn)
    input: afbeeldingen vector
    output: classifier en accuarcy, ook een voorbeeld.
    '''
    ## SVM part ##
    clf = sklearn.svm.SVC(gamma=0.001,C=100)
    clf.fit(x,y)
    p = x[-1:]#.reshape(-1, 1)

    ## accuacy quick check ##
    guess = []
    for x,y in zip(list(y_val),list(clf.predict(x_val))):
        if x == y:
            guess.append("True")
        else:
            guess.append("False")
    acc = guess.count("True")/len(guess)
    print('Accuarcy = ',acc)
    # plt.show()
    return clf

def auc_svm(X_train,y_train,X_test,y_test, plot = True):
    '''
    Simple support vector machine, setting are customizable, but didn't changes them during the experiment.
    input: image vector or feature vector
    output: classifier and accuarcy
    '''
    #normalize
    X_train = X_train/255
    X_test = X_test/255

    #warming message, large svm sets can take almost one hour to complete, but most are ready in 10min.
    print('SVM is busy... this may take a while: DO NOT CLOSE PROGRAM')

    # Learn to predict each class against the other # with: gamma=0.1,C=100,
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True)) 
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = y_train.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    AUC = auc(fpr["macro"], tpr["macro"])

    #plots figure (for testing)
    if plot:
        plt.figure()
        lw = 2

        plt.plot(fpr[2], tpr[2], color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")

        plt.show()
    return AUC