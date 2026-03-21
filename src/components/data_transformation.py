import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

## Config Class
@dataclass  ## auto constructor bnata hai clean code
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts", "preprocessor.pkl") # safe path bnata hai

#main Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig() 

    def get_data_transformer_object(self): # preproessing func

        '''This function is responsible for data transformation
        '''

        try:
            numerical_columns=['reading_score', 'writing_score']
            categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False)) ## ohe sparse matrix deta hai and scaler mean substract krta hai 
                ]
            )

            logging.info(f"Catagorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)

                ]
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path): # main transformation func

        try:
            train_df=pd.read_csv(train_path) ## data load
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            Preprocessing_obj=self.get_data_transformer_object()   # pipeline get

            target_column_name="math_score"
            numerical_columns=['reading_score', 'writing_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name])  # split features and target
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=Preprocessing_obj.fit_transform(input_feature_train_df) ## apply transformation
            input_feature_test_arr=Preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]  ## Combine X & y for train and test
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,   # save preprocessor by func call from utils.py
                obj=Preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path   # return jo aage model traing me jayega
            )
        except Exception as e:
            raise CustomException(e,sys)