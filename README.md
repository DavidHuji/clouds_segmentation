# clouds_segmentation
segmentation of satellite images of clouds into three types Open MCC, Closed MC and others.


### Training and Evaluation
In order to train and evaluate, simply run python main.py, you can change or tune the configuration in the macros.py file before.
At the first run, you will need to updates the paths of the data directories, the directories structure of the data should be as following: \
main_data_dir- \
&nbsp;&nbsp;-Train\
        &nbsp;&nbsp;&nbsp;&nbsp;-Images \
        &nbsp;&nbsp;&nbsp;&nbsp;-Masks \
    &nbsp;&nbsp;-Test \
        &nbsp;&nbsp;&nbsp;&nbsp;-Images \
        &nbsp;&nbsp;&nbsp;&nbsp;-Masks \
    &nbsp;&nbsp;-Valid     \
        &nbsp;&nbsp;&nbsp;&nbsp;-Images \
        &nbsp;&nbsp;&nbsp;&nbsp;-Masks 
       
      
The final results of confusion matrixes per set, weights, and log of test\train curves of loss and accuracy will be in the output directory that its name is just a timestemp.        


### Create Patches
In order to plit the data from full images to many small paches, you can use the patches_maker.py file by simply edit its arguemnts (of paths and sizes) at the head of the file and run it.
