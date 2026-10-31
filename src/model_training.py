import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.config_path import *
from src.custom_function import read_yaml, load_data

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)
            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)
            
            X_train = train_df.drop(columns=["Class"])
            y_train = train_df["Class"]
            X_test = test_df.drop(columns=["Class"])
            y_test = test_df["Class"]
            
            logger.info("Data split successfully for model training")
            return X_train, y_train, X_test, y_test 
        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise CustomException("Failed to load data", e)
    
    def model_training(self, X_train, Y_train):
        try:
            logger.info("Initializing model")
        
            model = LogisticRegression(max_iter=1000)
            logger.info("Starting model training")
            model.fit(X_train, Y_train)
            logger.info("Model training completed successfully")
            return model
        except Exception as e:
            logger.error(f"Error while training model: {e}")
            raise CustomException("Failed model training", e)

    def evaluate_model(self, model, X_test, Y_test):
        try:
            logger.info("Evaluating model")
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(Y_test, y_pred)
            precision = precision_score(Y_test, y_pred)
            recall = recall_score(Y_test, y_pred)
            f1 = f1_score(Y_test, y_pred)
            
            logger.info(f"Accuracy Score : {accuracy}")
            logger.info(f"Precision Score : {precision}")
            logger.info(f"Recall Score : {recall}")
            logger.info(f"F1 Score : {f1}")
            
            return {
                "recall_score": recall,
                "precision_score": precision,
                "accuracy_score": accuracy,
                "f1_score": f1
            }
        except Exception as e:
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException("Failed to evaluate model", e)

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info("Saving the model locally")
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved locally to {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error while saving model: {e}")
            raise CustomException("Failed to save model", e)
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training pipeline")
                
              
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                
                
                logger.info("Logging datasets to mlflow")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                
               
                model = self.model_training(X_train, y_train)
                
                self.save_model(model)
                metrics = self.evaluate_model(model, X_test, y_test)
                logger.info("Logging model parameters and metrics to mlflow")
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                
               
                mlflow.sklearn.log_model(model, "logistic_regression_model")
                logger.info("Model pipeline completed and tracked on MLflow successfully!")
                
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException("Failed during model training pipeline", e)

if __name__ == "__main__":
    trainer = ModelTraining(PREPROCESS_TRAIN_DATA_PATH, PREPROCESS_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()
