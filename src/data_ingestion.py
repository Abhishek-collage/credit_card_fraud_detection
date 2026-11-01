import os
import pandas as pd
from src.logger import get_logger 
from src.custom_exception import CustomException
from config.config_path import *
from src.custom_function import read_yaml
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.db_name = self.config["db_name"]
        self.collection = self.config["collection"]
        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"Data Ingestion started with {self.collection}is {self.db_name}")

    def download_csv_from_mongodb(self):
        try:
            client = MongoClient(os.getenv("uri"))
            logger.info("Successfully connected to mongodb")
            database = client[self.db_name]
            collection = database[self.collection]
            all_recordes = list(collection.find({}))
            df = pd.DataFrame(all_recordes)
            if '_id' in df.columns:
                df.drop(columns=['_id'], inplace=True)
            
            df.to_csv(RAW_FILE_PATH, index=False)
            logger.info("Successfully dataset saved")
        except Exception as e:
            logger.error("Error while downloading datasert")
            raise CustomException("Failed to split data into training  and test sets", e)
    def run(self):
        try:
            logger.info("Starting data ingestion process ")
            self.download_csv_from_mongodb()
            logger.info("Data ingestion  completed successfully")
        except CustomException as e:
            logger.error(f"CustomException :{str(e)}")

if __name__ == "__main__":
    config_path = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config_path)
    data_ingestion.run()