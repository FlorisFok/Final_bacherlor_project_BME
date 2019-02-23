######################################################################################
# Floris Fok
# Final bacherlor project
#
# 2019 febuari
# Transfer learning from medical and non medical data sets to medical target data
#
# ENJOY
######################################################################################
# Functions concering Data handeling
######################################################################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import sklearn
from sklearn import datasets, svm, metrics
import os
import random
import pandas
import time
import pickle

def import_melanoom(DIR, img_size_x,img_size_y, norm = False, color = False, classes = "two_combined"):
    '''
    Import function for melanoom, uses the folder structures as gotten from https://challenge2018.isic-archive.com/
    input: Folder location, image dimensions, normalize boolean, color boolean and Custom sub data classes
    output: images and labels.
    '''

    #data sizes
    TWO = 1746
    THREE = 2000

    try: 
        #searches folder for csv
        first_folder = os.listdir(DIR)
        data_frame = pandas.read_csv(os.path.join(DIR,first_folder[1]))
        data_frame = data_frame.set_index("image_id")
        #creates image files path.
        data_dir = os.path.join(DIR,first_folder[0])
        print('Found directory')
    except:
        #if not able to open one of the above, return error and empty strings
        print("Directory path not found")
        return [],[]

    # Set placeholders
    training_data = list()
    target_data = list()

    # selecets folders, sets timer for loading bar
    folders = os.listdir(data_dir)
    start = time.time()

    i = 0
    for img in folders:
        #skip superpixel files
        if 'superpixels' in img:
            continue

        #extracted label
        class_num = data_frame.loc[img[0:-4],:]
        #Choose label format, own custom sub data structure
        if classes == "two":
            size = TWO
            if class_num[0] == 1:
                class_num = [0,1]
            elif class_num[1] == 1:
                continue
                class_num = [0,1]
            else:
                class_num = [1,0]
            col_dim = 2
        elif classes == "three":
            size = THREE
            if class_num[0] == 1:
                class_num = [0,1,0]
            elif class_num[1] == 1:
                class_num = [0,0,1]
            else:
                class_num = [1,0,0]
            col_dim = 3
        elif classes == "two_combined":
            size = THREE
            if class_num[0] == 1:
                class_num = [0,1]
            elif class_num[1] == 1:
                class_num = [0,1]
            else:
                class_num = [1,0]
            col_dim = 2
        #extract image if pixel are not corrupted, either in color or black/white, finally resize
        try:
            if color:
                D = 3
                img_array = cv2.imread(os.path.join(data_dir,img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array,(img_size_x, img_size_y))
            else:
                img_array = cv2.imread(os.path.join(data_dir,img), cv2.IMREAD_GRAYSCALE)
                D = 1
                new_array = cv2.resize(img_array,(img_size_x, img_size_y))
            new_array = cv2.resize(img_array,(img_size_x, img_size_y))
            training_data.append(new_array)
            target_data.append(class_num)
        except Exception as e:
            pass
        #Funny load fucntion that visualizes the process
        loading(size,i,start, "melanoom data import")
        i+=1
    print('\n')

    #Make numpy array from data
    x = np.array(training_data).reshape(-1,img_size_x, img_size_y,D)
    y = np.array(target_data).reshape(-1,col_dim)

    #Normalize, if preferred
    if type(norm) != bool:
        print("please enter 'boolean' for norm(alization)")
    elif norm:
        x = x/ 255.0

    #Update on dataset size.
    print(f"This melanoom dataset contains the following: \nTotal length Dataset = {len(x)}")
    return x,y 

def import_kaggleDR(path, img_size_x, img_size_y, norm, color = False, limit = None):
    """
    Import fucntion for Retina data as provided by Kaggle.com
    input: Folder location, image dimensions, normalize boolean, color boolean and a size limit
    output: images and labels.
    """ 

    #format paths
    target_path = path + r"\trainLabels.csv\trainLabels.csv"
    train_path = path + r"\train"
    data_frame = pandas.read_csv(target_path)
    data_frame = data_frame.set_index("image")
    #find folders
    try:
        os_dir_list = os.listdir(train_path)
        print('Found directory, start import data')
    except:
        print("Directory path not found")
        return [],[]
    
    #set lists, and limit, if needed (testing purposes)
    training_data = list()
    target_data = list()
    if limit:
        size = limit
    else:
        size = len(os_dir_list)

    #start timer for loading screen
    start = time.time()
    i = 0

    #find images by folder, and adding label, resize to preference
    for img_name in os_dir_list:
        img_data = data_frame.loc[img_name[:-5],:]
        try:
            if color:
                img_array = cv2.imread(os.path.join(train_path,img_name), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                D = 3
            else:
                img_array = cv2.imread(os.path.join(train_path,img_name), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                D = 1
        except Exception as e:
            pass
        training_data.append(new_array)
        target_data.append(int(img_data))
        loading(size,i,start, "KaggleDR data import")
        i+=1
        #breaks loop at limit
        if limit and i > limit-1:
            break
    print('\n')

    #makes numpy arrays from data
    x = np.array(training_data).reshape(-1,img_size_x, img_size_y,D)
    y = np.array(target_data).reshape(-1)

    #Restores numbers to 0 and 1 notation of classes
    y,c = make_pre_train_classes(y, numb_classes = 5)

    #normalize if preffered.
    if type(norm) != bool:
        print("please enter 'boolean' for norm(alization)")
    elif norm:
        x = x/ 255.0

    print(f"This melanoom dataset contains the following: \nTotal length Dataset = {len(x)}")
    return x,y 

def import_dogcat(path, img_size_x,img_size_y, norm, color):
    """
    Import fucntion for dogcat data as provided by microsoft
    input: Folder location, image dimensions, normalize boolean, color boolean
    output: images and labels.
    """ 
    ## Works almost the same ##
    try:
        cat = list(os.listdir(path))
        print('Directory found')
    except:
        print('Directory not Found')


    training_data = list()
    training_class = list()
    
    for category in cat:
        class_num = cat.index(category)
        path2 = os.path.join(path, category)
        start = time.time()
        i = 0
        size = len(list(os.listdir(path2)))
        for img in os.listdir(path2):
            try:
                if color:
                    D = 3
                    img_array = cv2.imread(os.path.join(path2,img), cv2.IMREAD_COLOR)
                    new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                    
                else:
                    D = 1
                    img_array = cv2.imread(os.path.join(path2,img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                training_data.append(new_array)
                if class_num == 0:
                    training_class.append([1,0])
                elif class_num == 1:
                    training_class.append([0,1])
            except Exception as e:
                pass
            loading(size,i,start, "Cat_Dog data import")
            i+=1
        print("\n")

    x = np.array(training_data).reshape(-1,img_size_x, img_size_y,D)
    y = np.array(training_class).reshape(-1,len(cat))

    if type(norm) != bool:
        print("please enter 'boolean' for norm(alization)")
    elif norm:
        x = x/ 255.0

    print(f"This Dog_Cat dataset contains the following: \nTotal length Dataset = {len(x)} ")
    return x, y

def import_chest(path, img_size_x,img_size_y, norm, color):
    """
    Import fucntion for Retina data as provided by Kaggle.com
    input: Folder location, image dimensions, normalize boolean, color boolean
    output: images and labels.
    """ 
    ## works almost the same
    try: 
        types = list(os.listdir(path))
        print('Directory found')
    except:
        print('Directory not Found')

    training_data = list()
    training_class = list()
    

    for type_set in types:
        path2 = os.path.join(path, type_set)
        cat = list(os.listdir(path2))

        for category in cat:
            class_num = cat.index(category)
            path3 = os.path.join(path2, category)
            start = time.time()
            i = 0
            size = len(list(os.listdir(path3)))
            for img in os.listdir(path3):
                try:
                    if color:
                        D = 3
                        img_array = cv2.imread(os.path.join(path3,img), cv2.IMREAD_COLOR)
                        new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                        
                    else:
                        D = 1
                        img_array = cv2.imread(os.path.join(path3,img), cv2.IMREAD_GRAYSCALE)
                        new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                        print("done")
                    training_data.append(new_array)
                    if class_num == 0:
                        training_class.append([1,0])
                    elif class_num == 1:
                        training_class.append([0,1])
                except Exception as e:
                    pass
                loading(size,i,start, "Chest data import")
                i+=1

            print("\n")

    x = np.array(training_data).reshape(-1,img_size_x, img_size_y,D)
    y = np.array(training_class).reshape(-1,len(cat))

    if type(norm) != bool:
        print("please enter 'boolean' for norm(alization)")
    elif norm:
        x = x/ 255.0

    print(f"This Chest dataset contains the following: \nTotal length Dataset = {len(x)} ")
    return x, y

def import_blood(path, img_size_x,img_size_y, norm, color):
    """
    Import fucntion for Blood data as provided by Kaggle.com
    input: Folder location, image dimensions, normalize boolean, color boolean
    output: images and labels.
    """ 
    ## works almost the same
    try: 
        types = list(os.listdir(path))
        print('Directory found')
    except:
        print('Directory not Found')

    training_data = list()
    training_class = list()
    

    for type_set in types:
        path2 = os.path.join(path, type_set)
        cat = list(os.listdir(path2))

        for category in cat:
            class_num = cat.index(category)
            path3 = os.path.join(path2, category)
            start = time.time()
            i = 0
            size = len(list(os.listdir(path3)))
            for img in os.listdir(path3):
                try:
                    if color:
                        D = 3
                        img_array = cv2.imread(os.path.join(path3,img), cv2.IMREAD_COLOR)
                        new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                        
                    else:
                        D = 1
                        img_array = cv2.imread(os.path.join(path3,img), cv2.IMREAD_GRAYSCALE)
                        new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                        print("done")
                    training_data.append(new_array)

                    training_class.append(class_num)

                except Exception as e:
                    pass
                loading(size,i,start, "Chest data import")
                i+=1

            print("\n")

    x = np.array(training_data).reshape(-1,img_size_x, img_size_y,D)
    y = np.array(training_class).reshape(-1)

    y,c = make_pre_train_classes(y, numb_classes = 4)

    if type(norm) != bool:
        print("please enter 'boolean' for norm(alization)")
    elif norm:
        x = x/ 255.0

    print(f"This Chest dataset contains the following: \nTotal length Dataset = {len(x)} ")
    return x, y

def import_nat(path, img_size_x,img_size_y, norm, color):
    """
    Import fucntion for Blood data as provided by Kaggle.com
    input: Folder location, image dimensions, normalize boolean, color boolean
    output: images and labels.
    """ 
    ## works almost the same
    try: 
        types = list(os.listdir(path))
        print('Directory found')
    except:
        print('Directory not Found')

    training_data = list()
    training_class = list()
    
    cat = list(os.listdir(path))


    for category in cat:
        class_num = cat.index(category)
        path3 = os.path.join(path, category)
        start = time.time()
        i = 0
        size = len(list(os.listdir(path3)))
        for img in os.listdir(path3):
            try:
                if color:
                    D = 3
                    img_array = cv2.imread(os.path.join(path3,img), cv2.IMREAD_COLOR)
                    new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                    
                else:
                    D = 1
                    img_array = cv2.imread(os.path.join(path3,img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array,(img_size_x, img_size_y))
                    print("done")
                training_data.append(new_array)
                training_class.append(class_num)

            except Exception as e:
                pass
            loading(size,i,start, "Chest data import")
            i+=1
        print("\n")

    x = np.array(training_data).reshape(-1,img_size_x, img_size_y,D)
    y = np.array(training_class).reshape(-1)

    y,c = make_pre_train_classes(y, numb_classes = 8)

    if type(norm) != bool:
        print("please enter 'boolean' for norm(alization)")
    elif norm:
        x = x/ 255.0

    print(f"This Chest dataset contains the following: \nTotal length Dataset = {len(x)} ")
    return x, y

def make_pre_train_classes(Y, numb_classes = None):
    '''
    Makes number labels into zero and one labels
    input: list or array of numbers, number of classes
    output: class array of arrays of zero and ones.
    '''
    #checks if it's a list or array
    try:
        int(Y[0])
    except Exception as e:
        raise e
        return []
    
    #if class_numb is empty, calculate one.
    if not numb_classes:
        numb_classes = int(max(list(Y))-min(list(Y))+1)

    clas = list()

    #puts right amount of zeros in array with the one on the correct index
    for label in list(Y):
        new_list = numb_classes*[0]
        new_list[int(label)] = 1
        clas.append(new_list)
    #make numpy array
    clas_np = np.array(clas).reshape(-1,numb_classes)
    return clas_np

def get_data(params):
    """
    MEGA function to clean things up, returns data from the choosen params
    input: params(dict)
    output: x,y
    """
    #initializes random seed
    random.seed(params["RandomSeed"])

    # see if a pickle is available, doesn't require to import the data => faster
    try:
        print("Try to import pickle.")
        #selects right format for pickle laoding
        if params["Data"] == 'ISIC':
            zippy = list(pickle.load(open( f"{params['pickle_path']}{params['data_name']}.p", "rb" )))
        else:
            try:
                #small data set
                zippy = list(pickle.load(open( f"{params['pickle_path']}.p", "rb" )))
            except:
                #larger data set -- limit of pickle is 4gb
                zippy = list(pickle.load(open( f"{params['pickle_path']}_part1.p", "rb" )))
                zippy2 = list(pickle.load(open( f"{params['pickle_path']}_part2.p", "rb" )))
                zippy.extend(zippy2)
            ## could extend this if data is over 8gb!
        print("succeed to import pickle!")


    except:
        print("Failed to import pickle.")   
        ##Selects correct function for loading of data 
        if params["Data"] == 'CatDog' or params["Data"] == 'Breast':
            x,y = import_dogcat(params['file_path'], params['img_size_x'],params['img_size_y'], 
                norm = params["norm"], color = params["color"])
        elif params["Data"] == 'KaggleDR':
            x,y = import_kaggleDR(params['file_path'], params['img_size_x'],params['img_size_y'], 
                norm = params["norm"], color = params["color"])
        elif params["Data"] == 'ISIC':
            x,y = import_melanoom(params['file_path'], params['img_size_x'],params['img_size_y'],
                norm = params["norm"], color = params["color"],classes =params["data_name"])
        elif params["Data"] == 'Chest':
            x,y = import_chest(params['file_path'], params['img_size_x'],params['img_size_y'], 
                norm = params["norm"], color = params["color"])
        elif  params["Data"] == "Blood":
            x,y = import_blood(params['file_path'], params['img_size_x'],params['img_size_y'], 
                norm = params["norm"], color = params["color"])
        elif params["Data"] == "Nat":
            x,y = import_nat(params['file_path'], params['img_size_x'],params['img_size_y'], 
                norm = params["norm"], color = params["color"])

        #zips data so it can be stored and shuffled
        x = list(x)
        y = list(y)
        if params["Data"] == 'ISIC':
            zip_both = zip(x,y)
            pickle.dump( zip_both, open(f"{params['pickle_path']}{params['data_name']}.p", "wb" ))
        #depended on size --> one or two pickles, this is set for the demensions of 224x224x3, 
        # using larger dimensions could bring errors on savind--> adjust the size.
        elif len(y) < 15000:
            zip_both = zip(x,y)
            pickle.dump( zip_both, open( f"{params['pickle_path']}.p", "wb" ))
        else:
            zip1 = zip(x[int(len(y)/2):],y[int(len(y)/2):])
            zip2 = zip(x[:int(len(y)/2)],y[:int(len(y)/2)])
            pickle.dump( zip1, open( f"{params['pickle_path']}_part1.p", "wb" ))
            pickle.dump( zip2, open( f"{params['pickle_path']}_part2.p", "wb" ))
            zip_both = zip(x,y)
        zippy = list(zip_both)

    print("Shuffle and unzip data.")
    random.shuffle(zippy)
    x,y = zip(*zippy)

    #makes array and return them
    x = np.array(x)
    y = np.array(y)
    print("Data is loaded \n")

    return x,y

def count_classes(Y):
    '''
    Counts classes, no matter wich type of class notation it is, array of 1 and 0 or interger.
    input: class list
    output: class dict, {class:count}
    '''
    # checks if they are intergers ##
    
    back_to_num = list()
    list_classes = list(Y)
    for i in list_classes:
        back_to_num.append(list(i).index(1))
    # counts classes and returns a dict
    class_dict = dict()
    for n in back_to_num:
        if n in class_dict:
            class_dict[n] += 1
        else:
            class_dict[n] = 1
    return class_dict

def determen_weights(classes):
    """
    If classes are unevenly distrubuted we can add weights to the training to restore balance,
    This is done using this fucntion
    input: classes array
    output: dict with weights floats corresponding to class index
    """

    #count classes
    class_dict = count_classes(classes)

    #Make list from dict
    list_class_index = list(range(0,len(list(class_dict.keys()))))
    for key in class_dict.keys():
        list_class_index[key] = class_dict[key] 
    
    #devide class count per class by sum of classes
    weights = dict()
    for c in list_class_index:
        num = ((sum(list_class_index)/c))
        if num < 1:
            num = 1
        weights[list_class_index.index(c)] = float(num)

    return weights

def loading(size,i,start, name):
    """
    FUN loading annimation, no purpose.
    """
    stop = time.time()
    part = int(((i+1)/size)*20)
    loading_bar = part*'-'+(20-part)*' '
    print(f"{name}: {i+1}/{size}: [{loading_bar[0:10]}{part*5}%{loading_bar[10:20]}] elapsed time: {int(stop-start)}",end='\r')

def more_data(x,y,r):
    '''
    Data argmentation with keras
    input: image data, class data, amount of argmentations
    output: arrays of images and classes
    ''' 
    #itterates over the original and adds to new one.
    x_new = x
    y_new = y
    start = time.time()
    #makes r times new images
    for i in range(0,r):
        loading(r,i,start, "Data generator")
        datagen =  tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=360, fill_mode = "nearest")
        img=datagen.random_transform(x[i])
        y_new = np.concatenate((y_new, np.array(y[i]).reshape(1,y.shape[1])))
        x_new = np.concatenate((x_new, np.array(img).reshape(1,img.shape[0], img.shape[1],img.shape[2])))
    print("\n")
    return x_new,y_new

def equal_data(x,y):
    """
    If one class is larger then the others, this fucntion can one duplicate the less occuring classes
    input: image data, label data
    output: extended image data and extended label data
    """

    #count classes, determine the larges class
    d = count_classes(y)
    l = list()
    for key in d.keys():
        l.append(d[key])
    limit = max(l)

    #shuffle data
    zip_melanoom = zip(x,y)
    zippy = list(zip_melanoom)
    random.shuffle(zippy)
    x,y = zip(*zippy)
    x = np.array(x)
    y = np.array(y)
    
    # set blacklist of classes and datagenerator
    bad = list()
    datagen =  tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=360, fill_mode = "nearest")

    x_new = x
    y_new = y

    #irretare over entire dataset
    start = time.time()
    for i in range(0,len(y)):
        loading(len(y),i,start, "Data generator")
        
        # if the class is not lower than limit any more, add to black list and skip
        if not d[list(y[i]).index(1)] < limit:
            bad.append(list(y[i]))
            continue
        # if class is already on blacklist skip
        if list(y[i]).index(1) in bad:
            continue
        #if not on blackllist, argument image and extend to data.
        img=datagen.random_transform(x[i])
        y_new = np.concatenate((y_new, np.array(y[i]).reshape(1,y.shape[1])))
        x_new = np.concatenate((x_new, np.array(img).reshape(1,img.shape[0], img.shape[1],img.shape[2])))
        #check again if the class is within limits, needs to be done twice otherwise you mis front or back.
        d = count_classes(y_new)
        if not d[list(y[i]).index(1)] < limit:
            bad.append(list(y[i]).index(1))

    print("\n")

    return x_new,y_new

def equal_data_run(y_new,x_new):
    """
    Run equal data until all data is equal.... 
    """
    while True:
        d_old = count_classes(y_new)
        x_new, y_new = equal_data(x_new,y_new)
        d_new = count_classes(y_new)
        if d_old == d_new:
            break
    return x_new,y_new

def val_split(x,y, val_size):
    """
    Split data in two sets of valsize size.
    """
    return x[:val_size] ,y[:val_size] ,x[val_size:] ,y[val_size:]

def keep_class(x,y, classes):
    '''
    Elimentate certain classes (testing purpose)
    '''
    Y = []
    X = []
    for i,j in zip(list(y),list(x)):
        if list(i).index(1) in classes:
            Y.append(i)
            X.append(j)
    return X,Y

def equal_data_min(x,y):
    """
    Create equal data by removing images and labels only (not used)
    """
    d = count_classes(y)
    if type(d) == dict:
        l = list()
        for key in d.keys():
            l.append(d[key])
        limit = min(l)
    else:
        limit = min(d)

    back_to_num = list()
    for i in list(y):
        back_to_num.append(list(i).index(1))
    
    X = list()
    Y = list()
    d = dict()
    for n,img in zip(back_to_num,zip(x,y)):
        if n in d:
            if d[n] >= limit:
                continue
            d[n] += 1
            X.append(img[0])
            Y.append(img[1])
        else:
            d[n] = 1
            X.append(img[0])
            Y.append(img[1])
    return np.array(X),np.array(Y)