import argparse
import json
from utils import *

def create_gpt_training_data(groundtruth, prompts, output_file):
    ground_truth_ranking = normalized_fractional_ranking(groundtruth)
    labels = [int(r * 100.0) / 10.0 if r < 1.0 else 9.9 for r in ground_truth_ranking]
    
    training_data = []
    for prompt, label in zip(prompts, labels):
        training_data.append({
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": f"My answer is {label}."
                }
            ]
        })

    with open(output_file, 'w') as outfile:
        for data in training_data:
            outfile.write(json.dumps(data) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Create training data for GPT with groundtruth and GeoLLM prompts.")
    parser.add_argument('task_name', type=str, help='Name of the task')
    parser.add_argument('groundtruth_csv', type=str, help='Path to the CSV file containing groundtruth.')
    parser.add_argument('prompts_file', type=str, help='The file containing prompts')
    
    args = parser.parse_args()

    task = args.task
    groundtruth_csv = args.groundtruth_csv
    prompts_file = args.prompts_file

    df = pd.read_csv(groundtruth_csv)
    if 'Groundtruth' in df.columns:
        groundtruth = df['Groundtruth']
    else:
        raise ValueError("CSV file must contain 'Groundtruth' columns")

    prompts = load_geollm_prompts(prompts_file, task)

    if len(groundtruth) != len(prompts):
        raise ValueError("Number of groundtruth and prompts must be the same")

    output_file = f"gpt_training_data/{task}_training_data.jsonl"
    
    create_gpt_training_data(groundtruth, prompts, output_file)

if __name__ == "__main__":
    main()