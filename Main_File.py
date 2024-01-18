# Importing necessary packages
import numpy as np
import pandas as pd

from Input_Cleaning import Input_Cleaning
from Models import SVM_Model
from Models import SVM_Predictor
from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score

# Reading the data
Address_SurFail = 'C:/Users/nekooeimehr/AppData/Local/Programs/Python/Python35-32/SurgicalOutcomePrediction/Surgical Failure Input Data.xls'
xlsObj = pd.ExcelFile(Address_SurFail)
SurFail_data = xlsObj.parse('Sheet1', skiprows=7, index_col=None, na_values=['NA'])

# Preparing the Data
SurFail_data_Cleaned = Input_Cleaning(SurFail_data)

# Defining the output and input variables and scaling the inputs
Surgery_Outcome = SurFail_data_Cleaned['Sx Failure yes=1, no=0']
Input_Data = SurFail_data_Cleaned.iloc[:,1:]
Scaled_Input_Data = scale(Input_Data)

# Splitting the model to training set and testing set using leave-one-out CV
LOO_Indx = cross_validation.LeaveOneOut(len(Surgery_Outcome))
Prediction_Value_SVM = []
for train_indx, test_indx in LOO_Indx:
    # Preparing the Input Data (Data Balancing, Feature Selection, Noise Removal)
    (Prep_Train_Data, Prep_Surgery_Outcome, Selected_Features_Indx) = Input_Preparing(Scaled_Input_Data[train_indx], Surgery_Outcome[train_indx])

    # Building the first model
    svm_Tuned = SVM(Prep_Train_Data, Prep_Surgery_Outcome)

    # Predicting the test set using the built model
    Prep_Test_Data = Scaled_Input_Data.loc[test_indx, Selected_Features_Indx]
    Prediction_Value_SVM.append(SVM_Predictor(svm_Tuned, Prep_Test_Data))
    
FScore_SVM = f1_score(Surgery_Outcome, Prediction_Value_SVM, average=None)
Acc_SVM = accuracy_score(Surgery_Outcome, Prediction_Value_SVM, average=None)

