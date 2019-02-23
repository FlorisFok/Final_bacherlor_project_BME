######################################################################################
# Floris Fok
# Final bacherlor project
#
# 2019 febuari
# Transfer learning from medical and non medical data sets to medical target data
#
# ENJOY
######################################################################################
# Helper file for running experiments
# TIP: USE batch scripts
######################################################################################

import sys
#checks correct usage
if len(sys.argv) != 3:
    print("\nUSAGE: python run_batch.py iiii(i) 'run_style'")
    quit()

from AARunTarget import run_target
from AARunSource import run_source

#difine number string
arg = sys.argv[1]

#translate numbers to settings
model = ["imagenet","Chest","CatDog","KaggleDR","Nat",'None']
data = ['two','three','two_combined']#data[y]
style = ['FT', 'SVM']

#define argument numbers, just short to write
data_num = int(arg[0])
model_num  = int(arg[1])
style_num = int(arg[2])
random_num = int(arg[3])
#if a fifth argument is given, define it
try:
    sub_data_num = int(arg[4])
except:
    if data_num == 0:
        print("You need a fifth argument to run ISIC")

run_style = sys.argv[2]

# select paramset
if data_num ==0:
        params = {"Data":'ISIC',
                "data_name":data[sub_data_num],
                "style":style[style_num],
                "model":model[model_num],
                "file_path":r"C:\ISIC",
                "pickle_path":r"C:\pickles\save_melanoom_color_",
                 "model_path":{'Nat':r"C:\models\Epochs_50_Nat.json","KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
                "RandomSeed":random_num,
                "doc_path":r"C:\Users\Flori\Documents\GitHub\t",
                'img_size_x':224,
                'img_size_y':224,
                'norm':False,
                'color':True, 
                'pretrain':None, 
                "equal_data":False, 
                "shuffle":True, 
                "epochs":50, 
                "val_size":200,
                "test_size":400, 
                "Batch_size":32,
                "stop":'yes'
                }

elif data_num == 1:
        params = {"Data":'Nat',
                "style":'none',
                "model":'None',
                'file_path':r"C:\natural_images",
                'pickle_path':r"C:\pickles\Nat",
                'model_path':r"C:\models\Epochs_50_Nat.json",
                "RandomSeed":random_num,
                "doc_path":r"C:\Users\Flori\Documents\GitHub\t",
                'img_size_x':224,
                'img_size_y':224,
                'norm':False,
                'color':True, 
                'pretrain':None, 
                "equal_data":False, 
                "shuffle":True, 
                "epochs":50, 
                "val_size":500,
                "test_size":1000, 
                "Batch_size":32,
                "stop":'yes'
                }

elif data_num == 2:
        params = {"Data":'Blood',
                "data_name":None,
                "style":style[style_num],
                "model":model[model_num],
                "file_path":r"C:\blood-cells",
                "pickle_path":r"C:\pickles\Blood",
                "model_path":{"Nat":r"C:\models\Epochs_50_Nat.json","KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
                "RandomSeed":random_num,
                "doc_path":r"C:\Users\Flori\Documents\GitHub\t",
                'img_size_x':224,
                'img_size_y':224,
                'norm':False,
                'color':True, 
                'pretrain':None, 
                "equal_data":False, 
                "shuffle":True, 
                "epochs":50, 
                "val_size":1500,
                "test_size":2000, 
                "Batch_size":32,
                "stop":'yes'
                }
elif data_num == 3:
         params = {"Data":'Chest',
                "data_name":None,
                "style":style[style_num],
                "model":model[model_num],
                "file_path":r"C:\chest_xray",
                "pickle_path":r"C:\pickles\Chest_int",
                "model_path":{"Nat":r"C:\models\Epochs_50_Nat.json","KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
                "RandomSeed":random_num,
                "doc_path":r"C:\Users\Flori\Documents\GitHub\t",
                'img_size_x':224,
                'img_size_y':224,
                'norm':False,
                'color':True, 
                'pretrain':None, 
                "equal_data":False, 
                "shuffle":True, 
                "epochs":50 , 
                "val_size":600,
                "test_size":900, 
                "Batch_size":32,
                "stop":'yes'
                }
elif data_num == 4:
        params = {"Data":'KaggleDR',
                "data_name":None,
                "style":style[style_num],
                "model":model[model_num],
                "file_path":r"C:\KaggleDR",
                "pickle_path":r"C:\pickles\KaggleDR",
                "model_path":{"Nat":r"C:\models\Epochs_50_Nat.json","KaggleDR":r"C:\models\Epochs_50_kaggleDR.json","Chest":r"C:\models\Epochs_50_Chest.json", "CatDog":r"C:\models\Epochs_40_CatDog.json" },
                "RandomSeed":random_num,
                "doc_path":r"C:\Users\Flori\Documents\GitHub\t",
                'img_size_x':224,
                'img_size_y':224,
                'norm':False,
                'color':True, 
                'pretrain':None, 
                "equal_data":False, 
                "shuffle":True, 
                "epochs":50 , 
                "val_size":5000,
                "test_size":5000, 
                "Batch_size":32,
                "stop":'yes'
                }
elif data_num == 5:
    params = {"Data":'CatDog',
            "data_name":None,
            'file_path':r"C:\PetImages",
            "style":style[style_num],
            "model":model[model_num],
            'pickle_path':r"C:\pickles\CatDog",
            'model_path':r"C:\models\Epochs_",
            'doc_path':r"C:\Users\Flori\Documents\GitHub\t",
            "RandomSeed":random_num,
            'img_size_x':224,
            'img_size_y':224,
            'norm':True,
            'color':True,
            'pretrain':None,
            "equal_data":False, 
            "shuffle": True, 
            "epochs": 50 , 
            "val_size":3000,
            "test_size":5000, 
            "Batch_size": 32,
            "stop":'yes'
            }   

#run program
if run_style == 'source':
    run(params)
elif run_style == 'target':
    run_target(params)
else:
    print('Run Style Unknown')