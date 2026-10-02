from pathlib import Path
import os
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / 'data'
raw_data_path = data_dir / "raw"
interim_data_path = data_dir / "interim"
processed_data_path = data_dir / "processed" 
processed_dataset = processed_data_path/"features_static_after_split_TE.parquet"

# features
TARGET = 'isFraud'
TIME_COL = 'TransactionDT'
UID_COL = 'Pseudo_UID'
SEED = 42

# dagshub
load_dotenv()
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "anamish05")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "ieee-fraud-detection")
DAGSHUB_TOKEN=os.getenv("DAGSHUB_TOKEN")

