## All the common thing we use or common functionalities
import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path,obj):  ## For save the pickle file
    try:
        dir_path=os.path.dirname(file_path)  ## Full path se folder path nikalna
        os.makedirs(dir_path, exist_ok=True)  # folder create 

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  ## dump- save in this specific file path 

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train,y_train,X_test,y_test,models,params,cv=3,n_jobs=3,verbose=1,refit=False):
    try:
        report={}
        for i in range(len(models)):
            
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]
            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

    # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
        