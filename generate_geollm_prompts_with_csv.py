import argparse
import json
import pandas as pd
import requests
import concurrent.futures
import random
import math
import overpy
import time
from geopy.distance import geodesic
from tqdm import tqdm

MAXIMUM_NEARBY_PLACES = 10
MAXIMUM_RADIUS_IN_KM = 100

MIN_DELAY = 1
MAX_WORKERS = 10
TIMEOUT = 30

def calculate_initial_compass_bearing(lat1, lon1, lat2, lon2):
    if (lat1 == lat2) and (lon1 == lon2):
        return 0
    bearing = math.atan2(math.sin(lon2-lon1)*math.cos(lat2), math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(lon2-lon1))
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    return bearing

def bearing_to_compass(bearing):
    directions = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
    return directions[round(bearing/45) % 8]

def parse_places_data(result, lat, lon):
    places = []
    for node in result.nodes:
        if "place" in node.tags:
            place_name = node.tags.get("name", "n/a")
            place_lat, place_lon = float(node.lat), float(node.lon)
            distance = geodesic((lat, lon), (place_lat, place_lon)).km
            bearing = calculate_initial_compass_bearing(math.radians(lat), math.radians(lon), math.radians(place_lat), math.radians(place_lon))
            compass_direction = bearing_to_compass(bearing)
            places.append((place_name, distance, compass_direction))
    places.sort(key=lambda x: x[1])
    return places

def get_nearby_places(lat, lon):
    query = f"""
    (
        node(around:{MAXIMUM_RADIUS_IN_KM * 1000},{lat},{lon})["name"]["place"];
        >;
    );
    out meta;
    """
    result = overpy.Overpass().query(query)

    places = parse_places_data(result, lat, lon)

    nearby_places = ""
    for i, place in enumerate(places[:MAXIMUM_NEARBY_PLACES]):
        nearby_places += f"{place[1]:.1f} km {place[2]}: {place[0]}\n"

    return nearby_places

def get_address(lat, lon):
    nominatim_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1"

    response = requests.get(nominatim_url)
    data = json.loads(response.text)

    if 'error' in data:
        return None

    address = data['address']

    formatted_address = []
    for key in address:
        if "-" in key or "number" in key or "code" in key:
            continue
        formatted_address.append(address[key])

    return ', '.join(formatted_address)

def get_prompt(lat, lon):
    coordinates = f"({lat:.5f}, {lon:.5f})"
    address = get_address(lat, lon)
    nearby_places = get_nearby_places(lat, lon)
    prompt = f"Coordinates: {coordinates}\n\nAddress: \"{address}\"\n\nNearby Places:\n\"\n{nearby_places}\"\n\n<TASK> (On a Scale from 0.0 to 9.9): "
    return prompt

def get_prompt_executor(index, coordinates, prompts):
    try:
        lat, lon = coordinates[index]
        prompts[index] = get_prompt(lat, lon)
        print(f"Generated prompt {index + 1}")
    except Exception as e:
        print(f"Error while generating prompt {index + 1}: {e}")

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_prompts(coordinates, output_file=None):
    prompts = ["" for _ in range(len(coordinates))]
    prompt_indices = list(range(len(prompts)))

    while prompt_indices:
        failed_indices = []

        for chunk in chunks(prompt_indices, MAX_WORKERS):
            print("New Batch")

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(get_prompt_executor, index, coordinates, prompts): index for index in chunk}
                for future in concurrent.futures.as_completed(futures):
                    index = futures[future]
                    try:
                        future.result(timeout=TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        print(f"Generation for prompt {index + 1} did not complete within 30 seconds, rescheduling...")
                        failed_indices.append(index)
            
            time.sleep(MIN_DELAY)
            
            if output_file:
                with open(output_file, "w") as file:
                    for prompt in prompts:
                        if prompt:
                            file.write(json.dumps({"text": prompt}) + "\n")
        
        prompt_indices = failed_indices

    return prompts

def main():
    parser = argparse.ArgumentParser(description="Generate GeoLLM prompts based on coordinates.")
    parser.add_argument("coordinates_csv", type=str, help="Path to the CSV file containing coordinates.")
    parser.add_argument("--output_jsonl", type=str, help="Path to the output JSONL file. Defaults to the same name as the CSV file, located in the prompts/ folder.")

    args = parser.parse_args()

    coordinates_csv = args.coordinates_csv
    output_jsonl = args.output_jsonl if args.output_jsonl else "prompts/" + os.path.splitext(os.path.basename(coordinates_csv))[0] + ".jsonl"

    df = pd.read_csv(coordinates_csv)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        coordinates = list(zip(df['Latitude'], df['Longitude']))
    else:
        raise ValueError("CSV file must contain 'Latitude' and 'Longitude' columns")
    
    get_prompts(coordinates, output_jsonl)

if __name__ == "__main__":
    main()