import sys
import io
import time
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool

# --- WINDOWS UTF-8 ENCODING FIX (CRITICAL) ---
# Forces the console to use UTF-8 to prevent 'charmap' errors
# when printing special characters (emojis, etc.) on Windows.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# --- Local Modules ---
from src.core.connect_reddit import connect_reddit
from src.core.logger import setup_logger
from src.core.config_utils import get_config

# Pipeline Steps
from src.checkers.check_subreddits import main as check_main
from src.scrapers.subreddit_scraper import main as scrape_posts_main
from src.scrapers.comment_scraper import main as scrape_comments_main
from src.utils.cleaners import nlp_preprocess
from src.utils.save_csv import save_csv
from src.analyzers.data_aggregator import aggregate_main
from src.utils.kaggle_automator import KaggleAutomator

from src.analyzers.sentiment_analyzer import (
    load_sentiment_model,
    analyze_sentiment_batch,
    init_worker,
    process_chunk
)

# Suppress Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def load_processed_data(file_path, logger):
    """Helper function: Safely loads processed CSV files."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(f"⚠️ Found file {file_path.name} but it is empty.")
            return None
        return df
    except Exception as e:
        logger.error(f"❌ Failed to read {file_path.name}: {e}")
        return None


if __name__ == "__main__":

    # ==============================================================================
    # INITIALIZATION & CONFIGURATION
    # ==============================================================================
    today_str = datetime.today().strftime("%Y-%m-%d")
    logger = setup_logger()
    reddit = connect_reddit(logger)

    logger.info("Main process started. System initialized.")

    try:
        # Load Scraper Configurations
        post_limit = get_config("reddit_post_scraper", "post_limit", type=int)
        approve_limit = get_config("reddit_post_scraper", "post_comment_approve_limit", type=int)
        link_limit_conf = get_config("reddit_comment_scraper", "comment_link_limit", type=int)
        comment_limit = None if link_limit_conf == -1 else link_limit_conf
        scrape_till = datetime.utcnow() - timedelta(get_config("global", "comment_max_days", type=int))

        # Load Analysis Configurations
        # Lowercase to prevent case-sensitivity issues
        analysis_device = get_config("analysis", "device_type", fallback="cpu").lower()
        cpu_cores = get_config("analysis", "cpu_cores", type=int, fallback=4)

    except Exception as e:
        logger.critical(f"❌ Configuration Error: {e}")
        sys.exit(1)

    # --- PATH DEFINITIONS ---
    processed_dir = Path("data/processed/preprocessed_comments/")
    sentiment_dir = Path("data/processed/sentiment_scores/")
    live_dir = Path("data/dashboard/live/")
    timeseries_dir = Path("data/dashboard/time_series/")

    # Create directories if they don't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    sentiment_dir.mkdir(parents=True, exist_ok=True)

    # Define File Paths for Today
    CLEANED_FILE_PATH = processed_dir / f"{today_str}.csv"
    SENTIMENT_FILE_PATH = sentiment_dir / f"{today_str}.csv"

    # ==============================================================================
    # STEP 1-4: SCRAPING & PREPROCESSING PIPELINE
    # ==============================================================================

    processed_df = pd.DataFrame()

    # Check if we already have cleaned data for today
    if CLEANED_FILE_PATH.exists():
        logger.info(f"✅ Found existing cleaned data: {CLEANED_FILE_PATH.name}")
        processed_df = load_processed_data(CLEANED_FILE_PATH, logger)
        if processed_df is None:
            CLEANED_FILE_PATH.unlink(missing_ok=True)

    # If no data exists, run the scraper
    if processed_df is None or processed_df.empty:
        logger.info("⏳ Running Scraping Pipeline (Steps 1-4)...")

        # Step 1: Check and Update Subreddits
        logger.info("--- STEP 1: CHECK SUBREDDITS ---")
        subreddits_checked = check_main(pd.read_csv("assets/subreddits.csv"), reddit, logger)

        # Step 2: Scrape Reddit Posts
        logger.info("--- STEP 2: SCRAPE POSTS ---")
        posts_scraped = scrape_posts_main(
            subreddits_checked, logger, reddit, post_limit, approve_limit, scrape_till
        )

        # Step 3: Scrape Comments from Posts
        if not posts_scraped.empty:
            logger.info("--- STEP 3: SCRAPE COMMENTS ---")
            comments_scraped = scrape_comments_main(posts_scraped, logger, reddit, comment_limit)
        else:
            comments_scraped = pd.DataFrame()
            logger.warning("⚠️ No posts scraped. Skipping comments.")

        # Step 4: NLP Preprocessing (Cleaning)
        if not comments_scraped.empty:
            logger.info("--- STEP 4: PREPROCESSING ---")
            processed_df = nlp_preprocess(comments_scraped)

            if not processed_df.empty:
                save_csv(processed_df, logger, str(processed_dir))
                logger.info(f"✅ Preprocessing done. Saved to {CLEANED_FILE_PATH.name}")
            else:
                logger.error("❌ Preprocessing resulted in empty data.")
        else:
            logger.warning("⚠️ No comments to preprocess.")

    # ==============================================================================
    # STEP 5: SENTIMENT ANALYSIS (KAGGLE / GPU / CPU)
    # ==============================================================================

    sentiment_scores = pd.DataFrame()

    # Case 1: Analysis already done for today
    if SENTIMENT_FILE_PATH.exists():
        logger.info(f"✅ Found existing sentiment scores: {SENTIMENT_FILE_PATH.name}")
        sentiment_scores = load_processed_data(SENTIMENT_FILE_PATH, logger)

    # Case 2: Run Analysis on new data
    elif processed_df is not None and not processed_df.empty:
        logger.info("-" * 50)
        logger.info(f"🚀 STEP 5: STARTING SENTIMENT ANALYSIS (Mode: {analysis_device.upper()})")

        start_time = time.time()

        # --- MODE 1: KAGGLE AUTO (CLOUD AUTOMATION) ---
        if analysis_device == "kaggle_auto":
            try:
                # Initialize Automator
                automator = KaggleAutomator(logger, get_config)

                # 1. PRE-CLEANUP: Remove any leftover files from previous runs to prevent conflicts
                download_folder = Path("data/interchange/kaggle_download")
                if download_folder.exists():
                    shutil.rmtree(download_folder)
                    download_folder.mkdir(parents=True, exist_ok=True)
                    logger.info("🧹 Interchange folder cleaned before download.")

                # 2. Execute Kaggle Pipeline
                automator.prepare_and_upload_data(processed_df)  # Upload Dataset
                automator.push_and_run_kernel()  # Start Kernel
                automator.wait_for_completion()  # Wait for Finish

                # 3. Download Results (returns as DataFrame)
                sentiment_scores = automator.download_result()

                logger.info("✅ Kaggle Automation Cycle Complete.")

            except Exception as e:
                logger.critical(f"❌ Kaggle Automation Failed: {e}")
                sys.exit(1)

        # --- MODE 2: GPU (LOCAL) ---
        elif analysis_device == "gpu":
            try:
                logger.info("Loading Model to GPU...")
                model_pipeline, _, _ = load_sentiment_model()
                comments = processed_df['body'].tolist()
                logger.info(f"Analyzing {len(comments)} comments on GPU...")

                results = analyze_sentiment_batch(comments, model_pipeline)

                results_df = pd.DataFrame(results)
                processed_df = processed_df.reset_index(drop=True)
                sentiment_scores = pd.concat([processed_df, results_df], axis=1)

            except torch.OutOfMemoryError:
                logger.critical("❌ GPU Out Of Memory!")
                sys.exit(1)
            except Exception as e:
                logger.critical(f"❌ GPU Analysis Failed: {e}")
                sys.exit(1)

        # --- MODE 3: CPU (MULTIPROCESSING) ---
        else:
            logger.info(f"Splitting data across {cpu_cores} CPU cores...")
            df_chunks = np.array_split(processed_df, cpu_cores)
            with Pool(processes=cpu_cores, initializer=init_worker) as pool:
                results_list = pool.map(process_chunk, df_chunks)
            sentiment_scores = pd.concat(results_list, ignore_index=True)

        # --- POST-ANALYSIS: SAVE & CLEANUP ---
        duration = (time.time() - start_time) / 60
        logger.info(f"✅ Analysis Complete in {duration:.2f} minutes.")

        if not sentiment_scores.empty:
            # 1. Save to Permanent Storage (Processed Folder)
            sentiment_scores.to_csv(SENTIMENT_FILE_PATH, index=False)
            logger.info(f"💾 Sentiment scores saved to PERMANENT location: {SENTIMENT_FILE_PATH}")

            # 2. Cleanup Temporary Interchange File
            try:
                temp_file = Path("data/interchange/kaggle_download/results.csv")
                if temp_file.exists():
                    temp_file.unlink()
                    logger.info("🧹 Interchange temporary file removed.")
            except:
                pass

    else:
        logger.warning("⚠️ No data available for sentiment analysis.")

    # ==============================================================================
    # STEP 6: AGGREGATION & DASHBOARD UPDATE
    # ==============================================================================

    if not sentiment_scores.empty:
        logger.info("--- STEP 6: AGGREGATING SCORES ---")
        aggregated_scores = aggregate_main(sentiment_scores)

        if not aggregated_scores.empty:
            # Ensure directories exist
            live_dir.mkdir(parents=True, exist_ok=True)
            timeseries_dir.mkdir(parents=True, exist_ok=True)

            # Archive old dashboard data
            for old_file in live_dir.glob("*.csv"):
                try:
                    shutil.move(str(old_file), str(timeseries_dir / old_file.name))
                    logger.info(f"📦 Archived: {old_file.name}")
                except Exception as e:
                    logger.error(f"Failed to archive {old_file.name}: {e}")

            # Save new dashboard data (Live)
            new_agg_path = live_dir / f"{today_str}.csv"
            aggregated_scores.to_csv(new_agg_path, index=False)
            logger.info(f"✅ Dashboard Data Updated: {new_agg_path}")
        else:
            logger.warning("⚠️ Aggregation produced empty result.")
    else:
        logger.warning("⚠️ Skipping aggregation (No sentiment scores).")

    logger.info("🎉 PIPELINE FINISHED SUCCESSFULLY.")