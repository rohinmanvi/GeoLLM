import argparse
import pandas as pd
from utils import *
from scipy.stats import spearmanr

def calculate_spearman_correlation(coordinates, predictions, groundtruth_tif):
    groundtruth = [extract_data(lat, lon, groundtruth_tif) for lat, lon in coordinates]
    corr, _ = spearmanr(predictions, groundtruth)
    return corr

def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("predictions_csv", type=str, help="Path to the CSV file containing coordinates.")
    parser.add_argument("groundtruth_tif", type=str, help="Path to the groundtruth tif file.")

    args = parser.parse_args()

    predictions_csv = args.predictions_csv
    groundtruth_tif = args.groundtruth_tif

    df = pd.read_csv(predictions_csv)
    if 'Latitude' in df.columns and 'Longitude' in df.columns and 'Predictions' in df.columns:
        coordinates = list(zip(df['Latitude'], df['Longitude']))
        predictions = df['Predictions']
    else:
        raise ValueError("CSV file must contain 'Latitude', 'Longitude', and 'Predictions' columns.")

    corr = calculate_spearman_correlation(coordinates, predictions, groundtruth_tif)

    print(f"Spearman correlation: {corr:.2f}")

if __name__ == "__main__":
    main()