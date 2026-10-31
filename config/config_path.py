import os

########################### DATA INGESTION #########################

RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR,"raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR,"train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR,"test.csv")

CONFIG_PATH = "config/config.yaml"


######################## DATA PROCESSING ########################

PREPROCESS_DIR = "artifacts/processed"
PREPROCESS_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR,"processed_train.csv")
PREPROCESS_TEST_DATA_PATH = os.path.join(PROCESSED_DIR,"processed_test.csv")


####################### MODEL TRAINING #################
MODEL_OUTPUT_PATH = "artifacts/models"