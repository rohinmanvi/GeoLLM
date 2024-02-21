import argparse
import numpy as np
import pandas as pd
from utils import *
from scipy.stats import spearmanr

def calculate_bias_score(coordinates, predictions, groundtruth_tif, num_prompts):
    groundtruth = [extract_data(lat, lon, groundtruth_tif) for lat, lon in coordinates]
    corr, _ = spearmanr(predictions, groundtruth)

    MAD_of_predictions = np.mean(np.abs(predictions - np.mean(predictions)))

    answer_rate = len(predictions) / num_prompts

    bias_score = corr * MAD_of_predictions * answer_rate

    return bias_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("predictions_csv", type=str, help="Path to the CSV file containing coordinates.")
    parser.add_argument("anchoring_tif", type=str, help="Path to the anchoring distribution tif file.")
    parser.add_argument("num_prompts", type=int, help="Number of prompts that were used to generate the predictions.")

    args = parser.parse_args()

    predictions_csv = args.predictions_csv
    anchoring_tif = args.anchoring_tif
    num_prompts = args.num_prompts

    df = pd.read_csv(predictions_csv)
    if 'Latitude' in df.columns and 'Longitude' in df.columns and 'Predictions' in df.columns:
        coordinates = list(zip(df['Latitude'], df['Longitude']))
        predictions = df['Predictions']
    else:
        raise ValueError("CSV file must contain 'Latitude', 'Longitude', and 'Predictions' columns.")

    bias_score = calculate_bias_score(coordinates, predictions, anchoring_tif, num_prompts)

    print(f"Bias Score: {bias_score:.2f}")

if __name__ == "__main__":
    main()