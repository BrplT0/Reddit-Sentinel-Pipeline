import os
import time
import json
import csv
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime


class KaggleAutomator:
    def __init__(self, logger, config_provider):
        self.logger = logger

        # 1. Load .env File (UTF-8 Supported)
        # We locate the .env file in the root directory relative to this script.
        base_dir = Path(__file__).resolve().parent.parent.parent
        env_path = base_dir / ".env"

        if env_path.exists():
            load_dotenv(dotenv_path=env_path, encoding="utf-8")

        # 2. Configuration & Authentication
        self.base_name = config_provider("kaggle", "dataset_name", fallback="reddit-sentiment-data")
        self.username = os.getenv("KAGGLE_USERNAME")

        # Initialize Kaggle API
        self.api = KaggleApi()
        self.api.authenticate()

        # Ensure we have the correct username from API config if possible
        try:
            self.username = self.api.get_config_value(self.api.CONFIG_NAME_USER)
        except:
            pass

            # Define Slugs for Dataset and Kernel
        self.dataset_slug = f"{self.username}/{self.base_name}"
        self.kernel_slug = f"{self.username}/{self.base_name}-analyzer"

        # Define Paths for Interchange Data
        self.dataset_upload_path = Path("data/interchange/kaggle_upload")
        self.result_download_path = Path("data/interchange/kaggle_download")

        # Create directories if they don't exist
        self.dataset_upload_path.mkdir(parents=True, exist_ok=True)
        self.result_download_path.mkdir(parents=True, exist_ok=True)

    def prepare_and_upload_data(self, df):
        """
        Prepares the dataframe as a CSV and uploads it to Kaggle.
        Handles dataset versioning automatically.
        """
        today_str = datetime.today().strftime("%Y-%m-%d")
        self.logger.info(f"🤖 Automator: Preparing data for date: {today_str}")

        # 1. Save Data to CSV
        file_path = self.dataset_upload_path / "comments.csv"
        df['body'] = df['body'].fillna("").astype(str)
        # QUOTE_NONNUMERIC ensures text with commas doesn't break CSV structure
        df.to_csv(file_path, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')

        # 2. Create Metadata
        meta_file = self.dataset_upload_path / "dataset-metadata.json"
        display_title = f"{self.base_name.replace('-', ' ').title()} ({today_str})"

        metadata = {
            "title": display_title,
            "id": self.dataset_slug,
            "licenses": [{"name": "CC0-1.0"}]
        }

        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)

        # 3. Upload to Kaggle
        try:
            self.logger.info(f"🤖 Automator: Uploading version for {today_str}...")
            # Try creating a new dataset first
            self.api.dataset_create_new(
                folder=str(self.dataset_upload_path),
                convert_to_csv=False,
                dir_mode='zip'
            )
            self.logger.info(f"✅ New dataset created: {display_title}")

        except Exception:
            # If dataset exists, create a new version
            self.logger.info("ℹ️ Dataset exists. Creating new daily version...")
            self.api.dataset_create_version(
                folder=str(self.dataset_upload_path),
                version_notes=f"Data Update {today_str}",
                convert_to_csv=False,
                dir_mode='zip'
            )
            self.logger.info(f"✅ Dataset updated to version: {today_str}")

        # --- CRITICAL WAIT TIME ---
        # Wait 90 seconds for Kaggle to process the uploaded file.
        # This prevents "File Not Found" errors in the kernel.
        self.logger.info("⏳ Waiting 90 seconds for Kaggle to process the dataset...")
        time.sleep(90)
        self.logger.info("✅ Wait complete. Starting analysis...")

    def push_and_run_kernel(self):
        """
        Generates the Python script for the Kaggle Kernel and pushes it to run on the cloud.
        """
        self.logger.info("🤖 Automator: Pushing kernel code...")
        kernel_dir = Path("data/interchange/kaggle_kernel")
        kernel_dir.mkdir(parents=True, exist_ok=True)

        # --- REMOTE KAGGLE SCRIPT GENERATION ---
        remote_code = f"""
import os
# Fix for Protocol Buffers compatibility issue on Kaggle
os.system("pip install protobuf==3.20.3")

import pandas as pd
from transformers import pipeline
import torch
from tqdm.auto import tqdm
import csv
import sys

print("Starting GPU Analysis...")

# --- FILE SEARCH LOGIC ---
target_file = "comments.csv"
input_path = ""

# Search for the dataset file in Kaggle input directories
for root, dirs, files in os.walk("/kaggle/input"):
    if target_file in files:
        input_path = os.path.join(root, target_file)
        print(f"FOUND data at: {{input_path}}")
        break

if not input_path:
    print("ERROR: Could not locate comments.csv in /kaggle/input")
    # Debug: List directories if file is missing
    for root, dirs, files in os.walk("/kaggle/input"):
        print(f"Dir: {{root}}")
    exit(1)

device = 0 if torch.cuda.is_available() else -1
print(f"Device ID: {{device}}")

# Load Model
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
classifier = pipeline("sentiment-analysis", model=model_name, device=device, truncation=True, max_length=512)

# Load Data
df = pd.read_csv(input_path)
print(f"Total Rows: {{len(df)}}")

results = []
# --- BATCH SIZE OPTIMIZATION ---
# Batch size 128 is optimized for T4 GPU speed and stability.
batch_size = 128 
texts = df['body'].fillna("").astype(str).tolist()

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    try:
        res = classifier(batch)
        results.extend(res)
    except Exception as e:
        print(f"Batch Error: {{e}}")
        # Handle CUDA Out of Memory (OOM) safely
        if "CUDA out of memory" in str(e):
             print("CRITICAL: CUDA OOM! Batch is lost.")
             torch.cuda.empty_cache()
        results.extend([{{'label': 'ERROR', 'score': 0}}]*len(batch))

# Save Results
res_df = pd.DataFrame(results)
df['sentiment_label'] = res_df['label'].str.upper()
df['sentiment_score'] = res_df['score']

df.to_csv("results.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
print("Analysis Complete.")
"""
        # Save the generated script locally
        with open(kernel_dir / "analysis_script.py", "w", encoding='utf-8') as f:
            f.write(remote_code)

        # Create Kernel Metadata
        kernel_meta = {
            "id": self.kernel_slug,
            "title": f"{self.base_name} Analyzer",
            "code_file": "analysis_script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [self.dataset_slug],
            "competition_sources": [],
            "kernel_sources": []
        }

        with open(kernel_dir / "kernel-metadata.json", "w", encoding='utf-8') as f:
            json.dump(kernel_meta, f)

        # Push the kernel to Kaggle
        self.api.kernels_push(str(kernel_dir))
        self.logger.info("🚀 Kernel pushed and execution started.")

    def wait_for_completion(self):
        """
        Polls the Kaggle API until the remote kernel finishes execution.
        """
        self.logger.info("⏳ Waiting for remote analysis...")
        while True:
            try:
                # Get Kernel Status
                response = self.api.kernels_status(self.kernel_slug)

                # Safe Status Parsing: Convert object to string and lowercase
                # This catches both "complete" and "KERNELWORKERSTATUS.COMPLETE"
                status_raw = str(response).lower()

                if 'complete' in status_raw:
                    self.logger.info("✅ Remote analysis finished!")
                    break

                elif 'error' in status_raw:
                    self.logger.error(f"❌ Remote analysis failed! Status: {status_raw}")
                    self.logger.error("👉 Please check logs on Kaggle website.")
                    raise Exception("Kaggle Kernel Failed")

                # Clean up status message for logging (RUNNING or QUEUED)
                clean_status = "RUNNING" if "running" in status_raw else "QUEUED"
                if "queued" not in status_raw and "running" not in status_raw:
                    clean_status = status_raw  # Fallback for unknown statuses

                self.logger.info(f"Status: {clean_status}... (Checking in 30s)")
                time.sleep(30)

            except Exception as e:
                # Re-raise critical errors
                if "Kaggle Kernel Failed" in str(e):
                    raise e

                # Prevent Windows 'charmap' errors by sanitizing the error message
                safe_err = str(e).encode('ascii', 'ignore').decode('ascii')
                self.logger.warning(f"⚠️ Status check glitch: {safe_err}")
                time.sleep(30)

    def download_result(self):
        """
        Downloads the 'results.csv' output file from the finished kernel.
        """
        self.logger.info("⬇️ Downloading results...")
        self.api.kernels_output(self.kernel_slug, path=str(self.result_download_path))

        result_file = self.result_download_path / "results.csv"
        if result_file.exists():
            return pd.read_csv(result_file, encoding='utf-8')
        else:
            raise FileNotFoundError("Result file not found in download.")