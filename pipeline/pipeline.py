from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTraining
from src.common_functions import read_yaml
from config.config_path import *

if __name__=="__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
    preprocessor = DataPreprocessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PREPROCESS_DIR, CONFIG_PATH)
    preprocessor.process()
    trainer = ModelTraining(PREPROCESS_TRAIN_DATA_PATH, PREPROCESS_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run() 