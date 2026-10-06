from pathlib import Path
import os
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent
src_path=Path(__file__).resolve().parent

data_dir = root_dir / 'data'
raw_data_path = data_dir / "raw"
interim_data_path = data_dir / "interim"
processed_data_path = data_dir / "processed" 
processed_dataset = processed_data_path/"features_static_after_split_TE.parquet"

artifacts_path = src_path / "artifacts"

# features
TARGET = 'isFraud'
TIME_COL = 'TransactionDT'
UID_COL = 'Pseudo_UID'
SEED = 42
freq_cols = [
    'card1', 'card2', 'card3', 'card5',
    'addr1', 'addr2',
    'P_emaildomain', 'R_emaildomain',
    'DeviceInfo', 'id_30', 'id_31', 'id_33',
    'Pseudo_UID'
]

# dagshub
load_dotenv()
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "anamish05")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "ieee-fraud-detection")
DAGSHUB_TOKEN=os.getenv("DAGSHUB_TOKEN")

