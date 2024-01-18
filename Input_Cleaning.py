import numpy as np
import pandas as pd
        
def Input_Cleaning(SurFail_data):

    # Deleting the unnecessary features
    SurFail_data_Nec = SurFail_data.drop(['Stage-b','Parity-b','Age-b','BMI-b'], axis =1)  

    # Handeling the missing values by replacing them with median if continous and by mode if categorical
    SurFail_data_Nec.fillna(SurFail_data_Nec.median(), inplace = True)

    return(SurFail_data_Nec)
