######################################################################################
# Floris Fok
# Final bacherlor project
#
# 2019 febuari
# Transfer learning from medical and non medical data sets to medical target data
#
# ENJOY
######################################################################################
# Functions concering analysing of data, for testing and visualizing
# TIP: Use jupyter notebook for visualisation, the functions are not supported with
# any logic, 'some' manual work has to be done. I made all images with jupyter.
######################################################################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import sklearn
from sklearn import datasets, svm, metrics
from scipy.misc import imsave
import numpy as np

def conf_matrix(model, x_test,y_test):
    """
    plot a confusion matrix for analysing purposes
    input: model, test images, test labels
    oupput: (2 x plotted figure)
    """
    #make predictions
    y_pred = model.prediction(x_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

def Random_score(y):
    '''
    Get a score for randomness, takes steps equal to amount of classes, sees if it contains all the classes.
    Could set score limit
    input: classes
    output: score of given data
    '''
    ## checks if they are intergers based classes ##
    if type(Y[0]) != np.ndarray:
        back_to_num = list()
        for i in Y:
            back_to_num.append(list(i).index(1))

    score = 0
    numb_classes = max(y) - min(y) + 1 
    totaal = int(len(y) / numb_classes)
    score = 0
    for n in range(0, len(y[:totaal * numb_classes]), numb_classes):
        combi = (len(set(y[n:n + numb_classes])))
        score += combi / (totaal * numb_classes)
    print(score)
    return y, score

def count_classes():
    '''
    Counts classes, no matter wich type of class notation it is, array of 1 and 0 or interger.
    input: class list
    output: class dict, {class:count}
    '''
    ## checks if they are intergers ##
    if type(Y[0]) == np.ndarray:
        back_to_num = list()
        for i in Y:
            back_to_num.append(list(i).index(1))

    d = dict()
    for n in back_to_num:
        if n in d:
            d[n] += 1
        else:
            d[n] = 1
    return d


def plot_pre_train_result(H):
    # plot some features of the model that will help give guidance to result
        '''
    Bij gebruik van vele epochs zijn deze plots handig
    input: NN
    output: Graphs d.m.v. plt.show()
    '''
    # summarize history for accuracy
    plt.plot(H.history['acc'])
    plt.plot(H.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(H.history['loss'])
    plt.plot(H.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def vis_weights(filter_size, model, layer_name = 'block1_conv1',filter_index = 0):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    input_img = model.input
    img_width = filter_size
    img_height = img_width
    
    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    def nomralize(x):
        return x/(K.sqrt(K.mean(K.square(x))) + K.epsilon())

    # normalization trick: we normalize the gradient
    grads = nomralize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    step= 1.

    # we start from a gray image with some noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_width, img_height)) 
    else:
        input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data)* 20 + 128.


    # run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    # util function to convert a tensor into a valid image
    def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
    #     x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    img = input_img_data[0]
    img = deprocess_image(img)
    return img

def display_activation(layer_name = 'block1_conv1',model= applications.VGG16(include_top=False,weights='imagenet')):
    """
    Alternative for weights.
    """
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    col_size = int(int(((layer_dict[layer_name].output.shape[3])))**0.5)
    row_size = col_size
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    activation_index = 0
    for row in range(0,row_size):
        for col in range(0,col_size):
            img = vis_weights(50,model,layer_name,filter_index = activation_index)
            ax[row][col].imshow(img)
            activation_index += 1

#########################################################################################################
################    NOT USED, BUT USEFUL IN FURTHER PROJECCTS ###########################################

# def show_accuracy():
#     # check if prediction is rigth using few images (i guess 4)

# # def plot_layers(model.layer[0], x, y):
# #     filters = layer

# def plot_colorhis_dataset(x):
#     '''
#     Plots a histogram with amount of pixels per color of the data sets for RGB.
#     input: a dataset
#     output: figure
#     '''
#     colors = ("b", "g", "r")
#     fig = plt.figure()
#     plt.title("'Flattened' Color Histogram")
#     plt.xlabel("Bins")
#     plt.ylabel("# of Pixels")
#     features = []

#     # loop over the image channels
#     r = np.ndarray(shape=(256, 1), dtype=float)
#     g = np.ndarray(shape=(256, 1), dtype=float)
#     b = np.ndarray(shape=(256, 1), dtype=float)
#     for im in x:
#         chans = cv2.split(im)
#         for (chan, color) in zip(chans, colors):
#             # create a histogram for the current channel and
#             # concatenate the resulting histograms for each
#             # channel
#             hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
#             features.extend(hist)
#             if color == 'r':
#                 r+=hist
#             elif color == 'g':
#                 g+=hist
#             else:
#                 b+=hist

# #     r = (r/len(x)).astype(int) 
# #     b = (b/len(x)).astype(int) 
# #     g = (g/len(x)).astype(int)


#     for color,hist in zip(colors,[b,g,r]):
#         # plot the histogram
#         plt.plot(hist, color = color)
#         plt.xlim([0, 256])
#     return fig

# def plot_colorhis(image)
# '''
#     Plots a histogram with amount of pixels per color of a image for RGB.
#     input: a image
#     output: figure
#     '''
#     chans = cv2.split(image)
#     colors = ("b", "g", "r")
#     fig = plt.figure()
#     plt.title("'Flattened' Color Histogram")
#     plt.xlabel("Bins")
#     plt.ylabel("# of Pixels")
#     features = []
     
#     # loop over the image channels
#     for (chan, color) in zip(chans, colors):
#         # create a histogram for the current channel and
#         # concatenate the resulting histograms for each
#         # channel
#         hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
#         features.extend(hist)

#         # plot the histogram
#         plt.plot(hist, color = color)
#         plt.xlim([0, 256])
#     return fig

# def make_texture_his(img = cv2.imread('test.png')):
#     """
#     plots
#     """
#     METHOD = 'uniform'
#     radius = 3
#     n_point = 8*radius

#     def overlay_labels(image, lbp, labels):
#         mask = np.logical_or.reduce([lbp == each for each in labels])
#         return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


#     def highlight_bars(bars, indexes,c):
#         for i in indexes:
#             bars[i].set_facecolor(c)

#     image = rgb2gray(img).astype(int)
#     lbp = local_binary_pattern(image, n_points, radius, METHOD)


#     def hist(ax, lbp):
#         n_bins = int(lbp.max() + 1)
#         return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
#                        facecolor='0.5', label = ('','edge', 'flat', 'corner'))


#     # plot histograms of LBP of textures
#     fig, (ax_img, ax) = plt.subplots(nrows=2, ncols=1, figsize=(9, 6))
#     plt.gray()

#     titles = ('edge', 'flat', 'corner')
#     w = width = radius - 1
#     edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
#     flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
#     i_14 = n_points // 4            # 1/4th of the histogram
#     i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
#     corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
#                      list(range(i_34 - w, i_34 + w + 1)))

#     label_sets = (edge_labels, flat_labels, corner_labels)


#     ax_img.imshow(image)
    
#     color=['b', 'r', 'g']
    
# #     for ax, labels, name,c in zip(ax_hist, label_sets, titles,color):

#     counts, _, bars = hist(ax, lbp)
#     for c,indexes in zip(colors,label_sets):
# #         highlight_bars(bars, labels,c)
#         for i in indexes:
#             bars[i].set_facecolor(c)
#     ax.set_ylim(top=np.max(counts[:-1]))
#     ax.set_xlim(right=n_points + 2)
#     ax.set_title('Flat                   Corner                      Edge                   Corner                     Flat')
#     ax.set_ylabel('Percentage')
#     ax_img.axis('off')
#     return fig
