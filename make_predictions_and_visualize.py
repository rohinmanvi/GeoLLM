import argparse
from utils import *
import os
import json
import numpy as np
import pandas as pd
import math
import re
import time
import signal
import requests
import openai
import google.generativeai as genai
import folium

MIN_DELAY = 1.25

def handler(signum, frame):
    raise TimeoutError("Timeout occurred!")

def write_to_csv(latitudes, longitudes, predictions, file_path):
    df = pd.DataFrame({
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Predictions': predictions
    })
    df.to_csv(file_path, index=False)

def get_rating(completion):
    match = re.search(r"(\d+\.\d+)", completion)
    if not match:
        return None
    rating = float(match.group(0))
    return rating

def get_openai_prediction(api_key, model, prompt):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        max_tokens=10,
        temperature=0.0,
        logprobs=True,
        top_logprobs=5,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    completion = response.choices[0].message['content']
    most_probable = get_rating(completion)

    if most_probable is None:
        return None, None

    top_logprobs = response.choices[0].logprobs['content'][4]['top_logprobs']
    valid_items = [item for item in top_logprobs if item["token"].isdigit()]
    total_probability = sum(math.exp(item["logprob"]) for item in valid_items)
    expected_value = sum(int(item["token"]) * (math.exp(item["logprob"]) / total_probability) for item in valid_items)

    return completion, most_probable, expected_value

def get_google_prediction(api_key, model, prompt):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model)
    generation_config = genai.types.GenerationConfig(
        candidate_count=1,
        max_output_tokens=10,
        temperature=0
    )
    response = model.generate_content(prompt, generation_config=generation_config)
    completion = response.text
    most_probable = get_rating(completion)

    return completion, most_probable, None

def get_together_prediction(api_key, model, prompt):
    url = "https://api.together.xyz/v1/chat/completions"

    payload = {
        "model": model,
        "max_tokens": 10,
        "temperature": 0,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, json=payload, headers=headers)
    response = json.loads(response.text)

    completion = response['choices'][0]['message']['content']
    most_probable = get_rating(completion)

    return completion, most_probable, None

def plot_on_map(latitudes, longitudes, predicted, file_path):
    coordinates = list(zip(latitudes, longitudes))
    data = normalized_fractional_ranking(predicted)

    m = folium.Map(location=[20, 10], zoom_start=3.25, tiles='CartoDB positron')

    colormap = folium.LinearColormap(colors=['red', 'yellow', 'green'], vmin=0.0, vmax=1.0)
    colors = [colormap(val) for val in data]

    for coord, color in zip(coordinates, colors):
        folium.CircleMarker(
            location=coord,
            radius=7,
            color='none',
            fill=True,
            fill_color=color,
            fill_opacity=0.75
        ).add_to(m)

    m.add_child(colormap)

    m.save(file_path)

def run_task_for_data(model_api, model, task, prompt_file_path, api_key):
    signal.signal(signal.SIGALRM, handler)

    prompts = load_geollm_prompts(prompt_file_path, task)

    directory = "results"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    model_name = re.sub(r'[^a-zA-Z0-9_]', '_', model)
    task_name = re.sub(r'[^a-zA-Z0-9_]', '_', task)
    prompts_name = re.sub(r'[^a-zA-Z0-9_]', '_', prompt_file_path.split("/")[-1].split(".")[0])
    base_file_path = f"{directory}/{model_name}_{task_name}_{prompts_name}"

    i = 0
    latitudes = []
    longitudes = []
    predicted = []
    predicted_ev = []

    for prompt in prompts:
        try:
            i += 1

            lat, lon = get_coordinates(prompt)

            start_time = time.time()
            
            signal.alarm(20)

            if model_api == "openai":
                completion, most_probable, expected_value = get_openai_prediction(api_key, model, prompt)
            elif model_api == "google":
                completion, most_probable, expected_value = get_google_prediction(api_key, model, prompt)
            else:
                completion, most_probable, expected_value = get_together_prediction(api_key, model, prompt)

            signal.alarm(0)

            end_time = time.time()
            elapsed_time = end_time - start_time
            delay = max(MIN_DELAY - elapsed_time, 0)
            time.sleep(delay)

            print(f"PROMPT {i}:\n\n{prompt}\n\nCOMPLETION: {completion}\n")

            if most_probable is None:
                continue

            print(f"RATING: {most_probable}\n")

            latitudes.append(lat)
            longitudes.append(lon)

            predicted.append(most_probable)
            write_to_csv(latitudes, longitudes, predicted, f"{base_file_path}.csv")

            if expected_value is None:
                continue

            print(f"EXPECTED VALUE: {expected_value}\n\n")

            predicted_ev.append(expected_value)
            write_to_csv(latitudes, longitudes, predicted_ev, f"{base_file_path}_expected_value.csv")

        except Exception as e:
            print(f"Error encountered: {e}. Skipping this iteration.")
            continue

    plot_on_map(latitudes, longitudes, predicted, f"{base_file_path}.html")
    plot_on_map(latitudes, longitudes, predicted_ev, f"{base_file_path}_expected_value.html")

def main():
    parser = argparse.ArgumentParser(description='Run zero-shot predictions.')
    parser.add_argument('model_api', type=str, help='The API to use for predictions (openai, google, together)')
    parser.add_argument('api_key', type=str, help='The API key')
    parser.add_argument('model', type=str, help='The model to use for predictions')
    parser.add_argument('prompts_file', type=str, help='The file containing prompts')
    parser.add_argument('task', type=str, help='The task for predictions')

    args = parser.parse_args()

    model_api = args.model_api
    api_key = args.api_key
    model = args.model
    task = args.task
    prompt_file = args.prompts_file

    run_task_for_data(model_api, model, task, prompt_file, api_key)

if __name__ == "__main__":
    main()
