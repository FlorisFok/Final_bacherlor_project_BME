'''### LAPNOTES AUTOMATE --> write txt file with used params and datum and allllll ''' 
import matplotlib.pyplot as plt
import numpy as np
import time

def doc(par,res,H,doc_path):
    '''
    Saves results and params of the experiment
    Input: params(dict), results(dict), History of model, File path
    Output: Saved Fig and TXT.
    '''

    ## Get time for timestamp ##
    t = time.localtime()
    ## Make TXT ##
    # You should give your own title to it, this one wasn't perfect. A better idea is to include source and target data ##
    file_name= f"{t.tm_year}_{t.tm_mon}_{t.tm_mday}_RUNEXPERIMENT_RESULTS00{t.tm_hour}{t.tm_min}{t.tm_sec}.txt"
    f= open(r"{}ransfer-medical-bsc-2018\LAB\{}".format(doc_path,file_name),"w+")

    ## Make strings that contain all the info ##
    params = str()
    x = 1
    for item in par.keys():
        params = params + f"\n{x}." + str(item) + " = " + str(par[item])
        x+=1

    results = str()
    x = 1
    for item in res.keys():
        results = results + f"\n{x}." + str(item) + " = " + str(res[item])
        x+=1

    ## Make string of time stamp and owner ##
    # Your name should be placed here, or not...
    all_times = "Floris Fok \n" + str(t.tm_year)+"/" + str(t.tm_mon) +"/"+ str(t.tm_mday) +"      At: "+ str(t.tm_hour) +":" + str(t.tm_min)  +":"+ str(t.tm_sec)
    ## Finally write the TXT file
    f.write(f"{all_times}\n \n \n The params where: {params} \n \n \n The results where: {results}")

    ## IF a model was trained, the training of the model is captured in the figure ##
    if H != None:
        ## Make figure ##
        fig = plt.figure(num=None, figsize=(24, 12), dpi=80, facecolor='w', edgecolor='k')
        ## Make sub-figure ##
        plt.subplot(121)
        ## Plot values per epoch ##
        plt.plot(H.history['acc'])
        plt.plot(H.history['val_acc'])
        ## Make the figure readable ##
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        ## Make sub-figure ##
        plt.subplot(122)
        ## Plot values per epoch ##
        plt.plot(H.history['loss'])
        plt.plot(H.history['val_loss'])
        ## Make the figure readable ##
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        ## Save the figure as a whole ##
        # You should give your own title to it, this one wasn't perfect. A better idea is to include source and target data.
        fig_str = f"{doc_path[:-1]}figs\\{t.tm_year}_{t.tm_mon}_{t.tm_mday}_RUNEXPERIMENT_FigureLOSS&ACC0{t.tm_hour}{t.tm_min}.png"
        fig.savefig(fig_str, dpi=fig.dpi)
        fig_str2 = ' NONE '
        f.write(f"\nFigure name: \n{fig_str}"+"\n" + f"{fig_str2}")