import os
import numpy as np
import pandas as pd
from src.logger import get_logger
from src.custom_exception import  CustomException
from config.config_path import *
from src.custom_function import read_yaml, load_data
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        self.train_test_ratio = self.config["data_ingestion"]["train_ratio"]
    def split_data(self):
       
        try:
            logger.info("Starting the split data process")
            data = pd.read_csv(RAW_FILE_PATH)
            train_data, test_data = train_test_split(data, test_size=1-self.train_test_ratio, random_state=42, stratify=data["Class"] if "Class" in data.columns else None)

            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f" Train file saved {TRAIN_FILE_PATH}")
            logger.info(f"Test file saved {TEST_FILE_PATH}")
        except Exception as e:
            logger.error("Error while splitting ")
            raise CustomException(f"failed to split data training and testing", e)

   
    def save_data(self, df, file_path):
        try:
            logger.info("Saving data processed folder ")
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"error during saving data step {e}")
            raise CustomException("Error while saving data",e)
    def preprocess_data(self):
            try:
                logger.info("loding raw data from the directory")
                if not os.path.exists(self.train_path):
                    raise FileNotFoundError(f"Train file path does not exist")
                train_df = load_data(self.train_path)
                test_df = load_data(self.test_path)
                self.save_data(train_df, PREPROCESS_TRAIN_DATA_PATH)
                self.save_data(test_df, PREPROCESS_TEST_DATA_PATH)
            except Exception as e:
                

                logger.error(f"Error during preprocessing pipeline")
               
                raise CustomException("Error while data processing pipeline", e)


    def process(self):
        try:
            self.split_data()
            self.preprocess_data()


        except Exception as e:
            logger.error("Error during processing")
            import traceback
            print(traceback.format_exc())
            raise CustomException("Error while data preprocessing pipeline ", e)
if __name__ == "__main__":
    process = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PREPROCESS_DIR, CONFIG_PATH)
    process.process()