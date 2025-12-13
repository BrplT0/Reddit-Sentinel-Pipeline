import pandas as pd
import csv
import os
from transformers import pipeline
import torch
from src.core.config_utils import get_config

# --- GLOBAL VARIABLES (Required for Multiprocessing) ---
# These variables hold the model instance within worker processes.
# They are declared global to persist across chunk processing steps in the same worker.
model_pipeline_worker = None
model_label_map = None
current_model_name = None


def load_sentiment_model():
    """
    Loads the Hugging Face sentiment analysis pipeline.
    This logic is shared for both CPU workers and the Local GPU mode.
    """
    try:
        device_type = get_config("analysis", "device_type", fallback="cpu")
        model_name_config = get_config("analysis", "model_name", fallback="roberta")
    except Exception:
        print("WARN: Could not read config, defaulting to CPU and RoBERTa.")
        device_type = "cpu"
        model_name_config = "roberta"

    # Select Model Architecture
    if model_name_config.lower() == "distilbert":
        model_path = "distilbert-base-multilingual-cased"
    else:
        # Default: Twitter-XLM-RoBERTa (Recommended for Sentiment)
        model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    # Determine Device (GPU vs CPU)
    if device_type.lower() == "gpu":
        print(f"Config set to GPU. Using model: {model_path}")
        device_arg = 0  # 0 = First GPU
    else:
        print(f"Config set to CPU. Using model: {model_path}")
        device_arg = -1  # -1 = CPU

    # Initialize Pipeline
    model_pipeline = pipeline("sentiment-analysis",
                              model=model_path,
                              framework="pt",
                              truncation=True,
                              max_length=512,
                              device=device_arg)

    label_map = model_pipeline.model.config.id2label

    return model_pipeline, label_map, model_name_config


# ==============================================================================
# METHOD 1: CPU (MULTIPROCESSING) FUNCTIONS
# ==============================================================================

def init_worker():
    """
    Initializer function for multiprocessing pool workers.
    Loads the model once per worker process to avoid reloading it for every chunk.
    """
    global model_pipeline_worker
    global model_label_map
    global current_model_name

    print(f"Initializing sentiment model for worker process...")
    model_pipeline_worker, model_label_map, current_model_name = load_sentiment_model()
    print(f"Worker model loaded successfully: {current_model_name}")


def process_chunk(df_chunk):
    """
    Processes a specific chunk of the dataframe within a worker process.
    """
    global model_pipeline_worker
    global model_label_map
    global current_model_name

    # Safety check: Ensure model is loaded
    if model_pipeline_worker is None:
        print("WARN: Model was not loaded during init_worker, loading now...")
        init_worker()

    try:
        # 1. Handle Missing Values & Convert to String
        text_list = df_chunk['body'].fillna("").astype(str).tolist()

        # 2. Run Inference
        results = model_pipeline_worker(text_list, batch_size=64)
        results_df = pd.DataFrame(results)

        # 3. Standardize Labels (DistilBERT vs RoBERTa)
        if current_model_name and "distilbert" in current_model_name.lower():
            # DistilBERT usually outputs LABEL_0/LABEL_1
            distilbert_map = {
                'LABEL_0': 'NEGATIVE',
                'LABEL_1': 'POSITIVE'
            }
            results_df['sentiment_label'] = results_df['label'].replace(distilbert_map)
        else:
            # RoBERTa usually outputs negative/neutral/positive
            results_df['sentiment_label'] = results_df['label'].str.upper()

        # 4. Merge Results
        df_chunk = df_chunk.reset_index(drop=True)
        df_chunk['sentiment_label'] = results_df['sentiment_label']
        df_chunk['sentiment_score'] = results_df['score']

    except Exception as e:
        print(f"ERROR: Failed to process chunk: {e}")
        df_chunk['sentiment_label'] = 'ERROR'
        df_chunk['sentiment_score'] = 0.0

    return df_chunk


# ==============================================================================
# METHOD 2: GPU (LOCAL BATCH) FUNCTIONS
# ==============================================================================

def analyze_sentiment_batch(text_list, model):
    """
    Runs sentiment analysis on the entire list using the GPU.
    Includes fallback logic for CUDA Out Of Memory (OOM) errors.
    """
    batch_size_to_try = 128

    try:
        results = model(text_list, batch_size=batch_size_to_try)
    except torch.cuda.OutOfMemoryError:
        print(f"WARN: CUDA Out of Memory with batch_size={batch_size_to_try}. Retrying with batch_size=32...")

        # Clear Cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Retry with smaller batch
        results = model(text_list, batch_size=32)
    except Exception as e:
        print(f"ERROR during batch analysis: {e}")
        return []

    return results


# ==============================================================================
# METHOD 3: KAGGLE UTILITIES (EXPORT / IMPORT)
# ==============================================================================

def export_for_kaggle(df, output_path):
    """
    Prepares and saves data to CSV for Kaggle upload.
    Uses specific quoting options to prevent commas in text from breaking the CSV structure.
    """
    try:
        print(f"Exporting data for Kaggle to: {output_path}")

        # Ensure no NaN values in body
        df['body'] = df['body'].fillna("").astype(str)

        # quoting=csv.QUOTE_NONNUMERIC: Wraps all non-numeric fields (strings) in quotes.
        # This is critical for comments containing commas or newlines.
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

        print("✅ Export successful.")
        return True
    except Exception as e:
        print(f"❌ Failed to export data: {e}")
        return False


def load_from_kaggle_result(result_path):
    """
    Loads the analyzed CSV file downloaded from Kaggle.
    """
    if not os.path.exists(result_path):
        print(f"⚠️ Kaggle result file not found at: {result_path}")
        return None

    try:
        print(f"Loading Kaggle results from: {result_path}")
        df = pd.read_csv(result_path)

        # Validation: Check if sentiment columns exist
        if 'sentiment_label' not in df.columns or 'sentiment_score' not in df.columns:
            print("❌ Error: Loaded file is missing 'sentiment_label' or 'sentiment_score' columns.")
            return None

        print(f"✅ Loaded {len(df)} rows from Kaggle file.")
        return df
    except Exception as e:
        print(f"❌ Failed to load Kaggle results: {e}")
        return None