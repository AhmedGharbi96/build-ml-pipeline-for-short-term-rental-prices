#!/usr/bin/env python
"""
 Download from W&B the raw dataset and apply some basic data cleaning, exporting the
 result to a new artifact
"""
import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    logging.info(f"Downloading {args.input_artifact} passed.")

    logging.info("Starting data cleaning...")
    # Drop outliers
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Drop rows outside allowed longitude and latitude
    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[idx].copy()

    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])
    logging.info("Data cleaning finished.")

    # save the df in a tmp directory then push it to wandb
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir, "clean_sample.csv")
        df.to_csv(temp_path, index=False)
        logging.info("Uploading the cleaned dataframe to wandb")

        artifact = wandb.Artifact(
            name=args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )
        artifact.add_file(temp_path)

        logger.info("Logging artifact")
        run.log_artifact(artifact)
        artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=" A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the artifact to download from wandb",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the artifact to clean and upload to wand",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="""Type of the output artifact. This will be used to categorize the artifact
         in the W&B interface""",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A brief description of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price", type=float, help="Minimum price to keep", required=True
    )

    parser.add_argument(
        "--max_price", type=float, help="Maximum price to keep", required=True
    )

    args = parser.parse_args()

    go(args)
