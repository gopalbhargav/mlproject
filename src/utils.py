## All the common thing we use or common functionalities
import os
import sys
import numpy as np
import pandas as pd
import dill

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
    

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train) # Train model

    # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)