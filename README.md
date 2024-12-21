# MF-Attention-BiLSTM-CRF
A Novel Named Entity Recognition Model for Bridge Inspection Report with Multi-Features Fusion and Correction Method
 
# Python environment

 pytorch == 2.2.1

 ltp == 4.2.14
 
# Catalogue
    ├── ReadMe.md
    
    ├── data_aug.py    // Create training set, validation set, and test set files. And perform data augmentation on the training set
        
    ├── optimization_results_evaluation.py    //  To verified the model performanceon on test set after the model training. The correction results will print togther.
            
    ├── main.py    //  Model training.
    
    │   ├── substructure     
    
    │       ├── data  // Location of files such as training sets
 
# Direction for use

    1. Run **data_aug.py** to creat training set, validation set, and test set. Data augmentation will be performed on the training set in **data_aug.py** and the number of augmentations is optional.
    
    2. Run **main.py** to train the MF-Attention-BiLSTM-CRF model.
    
    3.实现指定版本的include、src、lib文件的拷贝
 
 
