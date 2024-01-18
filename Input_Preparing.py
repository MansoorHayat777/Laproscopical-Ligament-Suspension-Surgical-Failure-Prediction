from mifs import mifs
from imblearn.over_sampling import SMOTE

def Input_Preparing(Scaled_Input_Data, Surgery_Outcome, N_Feat):
    # Feature Selection
    MIFS = mifs.MutualInformationFeatureSelector(method='JMI', verbose=2, n_features = N_Feat)
    MIFS.fit(Scaled_Input_Data, Surgery_Outcome)
    Selected_Input_Data = Scaled_Input_Data.loc[:,MIFS.support_]

    # Balancing using SMOTE
    sm = SMOTE(kind='regular')
    Prep_Train_Data, Prep_Surgery_Outcome = sm.fit_sample(X, y)
    
    return(Prep_Train_Data, Prep_Surgery_Outcome, MIFS.support_)
    
    
