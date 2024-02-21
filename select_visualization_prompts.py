import argparse
from utils import *
from itertools import combinations
import math
import random
from tqdm import tqdm

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def select_spread_out_points_with_importance_sampling(points, populations, num_points=2000, approx_sample=1000):
    if len(points) <= num_points:
        return range(len(points))

    indices = list(range(len(points)))
    indices.sort(key=lambda i: populations[i], reverse=True)

    farthest_points_indices = set([0])

    interval = len(points) // num_points
    population_cap_index = interval

    with tqdm(total=num_points, desc="Selecting Farthest Points") as pbar:
        while len(farthest_points_indices) < num_points:
            max_dist = -1
            farthest_point_index = None

            populated_indices = indices[:population_cap_index] if population_cap_index < len(points) else indices
            sampled_indices = random.sample(populated_indices, approx_sample) if len(populated_indices) > approx_sample else populated_indices

            for i in sampled_indices:
                point = points[i]
                if i not in farthest_points_indices:
                    min_dist_to_set = min(distance(point, points[idx]) for idx in farthest_points_indices)
                    if min_dist_to_set > max_dist:
                        max_dist = min_dist_to_set
                        farthest_point_index = i

            farthest_points_indices.add(farthest_point_index)

            population_cap_index += interval
            pbar.update(1)

    return list(farthest_points_indices)

def main():
    parser = argparse.ArgumentParser(description="Select prompts for visualization")
    parser.add_argument("prompts_file", help="Input file path")
    parser.add_argument("output_file", help="Output file path")
    parser.add_argument("num_points", type=int, help="Number of points to select")
    parser.add_argument("regions", nargs="+", help="List of regions")

    args = parser.parse_args()

    num_points = args.num_points
    regions = args.regions
    output_file_path = args.output_file
    file_name = args.prompts_file
    population_data_file_path = "data/ppp_2020_1km_Aggregated.tif"

    with open(file_name, 'r') as infile:
        lines = infile.readlines()

    lines = [line for line in lines if any(region in line for region in regions)]
    coordinates = [get_coordinates(line) for line in lines]

    with tqdm(total=len(coordinates), desc="Extracting Data") as pbar:
        populations = []
        for lat, lon in coordinates:
            population = extract_data(lat, lon, population_data_file_path)
            populations.append(population)
            pbar.update(1)

    indices = select_spread_out_points_with_importance_sampling(coordinates, populations, num_points)
    result = [lines[index] for index in indices]

    random.shuffle(result)

    with open(output_file_path, 'w') as f:
        f.writelines(result)

if __name__ == "__main__":
    main()
