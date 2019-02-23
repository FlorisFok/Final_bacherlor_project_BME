# Welcome
This is the script of the final bacherlor project of Floris Fok. This script allows transfer learning with both medical and non-medical datasets. It measures performance with AUC and has some fucntions in AAAnalyse to visualize the filters. This will determine if medical source data is required when applying transfer learning on medical target data. 

Abstract and conclusion can be found below, a short explaination of how to use the script is given here. The whole script is supported by comments to explain what is happening, so everybody will be able to understand the script with some prior knowledge of python.

# Script
### How to use, first of all you need to download the correct datasets:
* [Cat & Dog](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
* [Natural Images](https://www.kaggle.com/prasunroy/natural-images)
* [Blood cells](https://www.kaggle.com/paultimothymooney/blood-cells/home)
* [Chest x-rays](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* [KaggleDR](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
* [ISIC](https://challenge.kitware.com/#challenge/n/ISIC_2017%3A_Skin_Lesion_Analysis_Towards_Melanoma_Detection)

### Download all py files from github

### Edit the params in your script to match the format, skip the xxx.
 params = {"Data":xxx,
         "data_name":xxx,
         "style":xxx,
         "model":xxx,
         "file_path":PATH OF DATA SET FOLDER,
         "pickle_path":PATH OF PICKLE,
         "model_path":{'Nat':PATH OF EACH MODEL IF YOU MADE THEM} OR MODEL PATH IF ITS SOURCE DATA,
         "RandomSeed":xxx,
         "doc_path":PATH OF WERE YOUR RESULTS WILL BE PRINTED,
         'img_size_x':xxx,
         'img_size_y':xxx,
         'norm':xxx,
         'color': xxx, 
         'pretrain': xxx, 
         "equal_data": xxx, 
         "shuffle": xxx, 
         "epochs": xxx, 
         "val_size": xxx,
         "test_size": xxx, 
         "Batch_size": xxx,
         "stop": xxx
         }

### run run_batch.py
use correct usage of "python run_batch.py iiii(i) run_style"
* First run source data scripts (with exception of imagenet, these weights are from the web)
* Secondly run the experiments


# Abstract
Normally, deep neural networks in medical image analysis take a large amount of data to be effective. Medical data sets, which are frequently smaller then other image data sets, are therefore more difficult to use. Transfer learning is a method that utilizes knowledge learned from the source data to help classify the target data. This method demonstrates to work effectively with small medical datasets. There is still, however, some ambiguity if transfer learning from non-medical data is effective for medical data. In this paper, we compare the performance of transfer learning from non-medical and medical source data to medical target data. To generalize the experiment, we experiment with two transfer learning methods, feature extraction and fine-tuning and compare them with conventional training.The area under the ROC (receiver operating characteristic) curve of the classified target data is used to compare the experiments, which functions as a measurement for performance. Beside analysing the results, we also visualizing the filters of convolutional neural network (CNN), which deepens the understanding of the results. It turns out, transfer learning with both medical and non-medial data on average improves performance. Fine tuning outperforms the other transfer learning method in this paper. In addition, the filters from a pre-trained CNN correlate with the performance of transfer learning and we obtained a better understanding of why certain models achieve better performance when using transfer learning. With this knowledge, proposing different data characteristics to be researched to extend this understanding further.

# Conclusion
Non-medical data is suitable for transfer learning with medical target data, especially when fine-tuning is applied. When comparing different transfer leaning methods, fine-tuning improves the score more than no fine-tuning. The success of transfer learning is more likely to be correlated with the amount of data seen by the model and the development of structural filters; then similarity in data sets. Future research should focus on more general characteristics of the dataset that can improve the performance of transfer learning.
