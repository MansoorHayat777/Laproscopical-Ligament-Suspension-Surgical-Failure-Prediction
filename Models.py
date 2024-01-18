import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVM
from sklearn.cross_validation import StratifiedKFold

def SVM_Model(Scaled_Input_Data, Output_Data):
    Grid_Dict = {"C": [1e-1,1e0, 1e1],"gamma": np.logspace(-2, 1, 3)}
    cv_Strat = cross_validation.StratifiedKFold(Output_Data, n_folds=3)
    svm_Tuned = GridSearchCV(SVC(kernel='rbf', gamma=0.1, tol = 0.05), cv=cv_Strat, param_grid=Grid_Dict)
    svm_Tuned.fit(Scaled_Input_Data, Output_Data)
    return(svm_Tuned)

def SVM_Predictor(svm_Tuned, Input_test_Data):
    Predicted_SVM = svm_Tuned.predict(Input_test_Data)
    return(Predicted_SVM)

    
