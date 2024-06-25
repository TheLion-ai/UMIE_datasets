"""
This script downloads all files from a specified S3 bucket to a local directory.

Usage:
    python download_files.py

Environment Variables:
    - S3_ENDPOINT: The endpoint URL of the S3 service.
    - S3_ACCESS_KEY: The access key for the S3 service.
    - S3_SECRET_KEY: The secret key for the S3 service.

Constants:
    - BUCKET_NAME (str): The name of the S3 bucket.
    - LOCAL_DIR (str): The local directory to download the files to.

Functions:
    - download_s3_folder(bucket_name, local_dir=None): Downloads all files from a specified S3 bucket to a local directory.

"""

import os

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BUCKET_NAME = "umie-tests"
LOCAL_DIR = "testing/test_dummy_data"

client = boto3.client(
    service_name="s3",
    endpoint_url=os.environ["S3_ENDPOINT"],
    aws_access_key_id=os.environ["S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    config=Config(signature_version="s3v4"),
)


def download_s3_folder(bucket_name: str, local_dir: str = None) -> None:
    """
    Download all files from a specified S3 bucket to a local directory.

    Args:
        bucket_name (str): The name of the S3 bucket.
        local_dir (str, optional): The local directory to download the files to. If not provided, the files will be downloaded to the current working directory.

    Returns:
        None
    """
    print(f"Downloading files from {bucket_name} to {local_dir}...")
    paginator = client.get_paginator("list_objects")
    pages = paginator.paginate(Bucket=bucket_name)

    for i, page in enumerate(pages):
        print(f"Downloading page {i + 1}...")
        for obj in tqdm(page["Contents"]):
            target = obj["Key"] if local_dir is None else os.path.join(local_dir, obj["Key"])
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj["Key"][-1] == "/":
                continue
            with open(target, "wb") as f:
                client.download_fileobj(BUCKET_NAME, obj["Key"], f)


if __name__ == "__main__":
    if os.path.exists(LOCAL_DIR):
        print(f"Directory {LOCAL_DIR} already exists.")
    else:
        download_s3_folder(BUCKET_NAME, LOCAL_DIR)
