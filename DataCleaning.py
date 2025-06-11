import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import boto3
from botocore.exceptions import NoCredentialsError
from abc import ABC, abstractmethod
import os

# ---- Interfaces ----
class DataLoader(ABC):
    @abstractmethod
    def load(self):
        pass

class DataSaver(ABC):
    @abstractmethod
    def save(self, df: pd.DataFrame, path: str):
        pass

class StorageUploader(ABC):
    @abstractmethod
    def upload_file(self, file_path: str, key: str) -> str:
        pass

    @abstractmethod
    def download_file(self, key: str, download_path: str) -> str:
        pass

# ---- Concrete Implementations ----
class CSVDataLoader(DataLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        return pd.read_csv(self.file_path)

class CSVDataSaver(DataSaver):
    def save(self, df: pd.DataFrame, path: str):
        df.to_csv(path, index=False)

class S3Uploader(StorageUploader):
    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name

    def upload_file(self, file_path: str, key: str) -> str:
        try:
            self.s3.upload_file(file_path, self.bucket_name, key)
            return f"Uploaded {file_path} to s3://{self.bucket_name}/{key}"
        except NoCredentialsError:
            return "AWS credentials not available."

    def download_file(self, key: str, download_path: str) -> str:
        try:
            self.s3.download_file(self.bucket_name, key, download_path)
            return f"Downloaded s3://{self.bucket_name}/{key} to {download_path}"
        except NoCredentialsError:
            return "AWS credentials not available."

# ---- Data Cleaning Logic ----
class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def handle_missing(self):
        self.df = self.df.dropna()
        return self

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self

    def remove_outliers(self, z_thresh=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs((self.df[numeric_cols] - self.df[numeric_cols].mean()) / self.df[numeric_cols].std())
        self.df = self.df[(z_scores < z_thresh).all(axis=1)]
        return self

    def correct_data_types(self):
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        return self

    def standardize_strings(self):
        str_cols = self.df.select_dtypes(include='object').columns
        for col in str_cols:
            self.df[col] = self.df[col].str.lower().str.strip()
        return self

    def scale_numeric(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            scaler = StandardScaler()
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        return self

    def get_cleaned_data(self):
        return self.df

# ---- Main method ----
def main():
    bucket = "your-bucket-name"  # <-- Change this
    s3_input_key = "input-data/raw_data.csv"  # <-- Change this
    local_input_path = "/tmp/raw_data.csv"
    local_output_path = "/tmp/cleaned_data.csv"
    s3_output_key = "processed-data/cleaned_data.csv"  # <-- Change this

    try:
        uploader = S3Uploader(bucket)

        print(uploader.download_file(s3_input_key, local_input_path))

        loader = CSVDataLoader(local_input_path)
        df = loader.load()

        cleaner = DataCleaner(df)
        cleaner.handle_missing().remove_duplicates().remove_outliers()
        cleaner.correct_data_types().standardize_strings().scale_numeric()
        cleaned_df = cleaner.get_cleaned_data()

        saver = CSVDataSaver()
        saver.save(cleaned_df, local_output_path)

        print(uploader.upload_file(local_output_path, s3_output_key))

        print("Data cleaning and upload completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
