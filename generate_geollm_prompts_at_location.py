import argparse
import numpy as np
import rasterio
from rasterio.transform import rowcol
from generate_geollm_prompts_with_csv import get_prompts
from select_visualization_prompts import select_spread_out_points_with_importance_sampling

def generate_prompts(bbox, num_prompts, output_file):
    with rasterio.open('data/ppp_2020_1km_Aggregated.tif') as src:
        data = src.read(1)
        transform = src.transform
        
        top_left = rowcol(transform, bbox[0], bbox[3])
        bottom_right = rowcol(transform, bbox[2], bbox[1])
        
        data_cropped = data[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        
        pop_flat_cropped = data_cropped.flatten()
        
        indices_cropped = np.nonzero(pop_flat_cropped > 0)[0]
        
        rows_cropped, cols_cropped = np.unravel_index(indices_cropped, data_cropped.shape)
        
        rows_adjusted = rows_cropped + top_left[0]
        cols_adjusted = cols_cropped + top_left[1]
        
        valid_coords = [tuple(reversed(rasterio.transform.xy(transform, row, col))) for row, col in zip(rows_adjusted, cols_adjusted)]
        populations = [data[row, col] for row, col in zip(rows_adjusted, cols_adjusted)]

    selected_indices = select_spread_out_points_with_importance_sampling(valid_coords, populations, num_prompts)
    valid_coords = [valid_coords[i] for i in selected_indices]
    
    get_prompts(valid_coords, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts at a specific location")
    parser.add_argument("output_file", type=str, help="Output file path")
    parser.add_argument("num_prompts", type=int, help="Number of prompts to generate")
    parser.add_argument("bbox", type=float, nargs=4, help="Bounding box coordinates (min_lat, min_lon, max_lat, max_lon)")
    
    args = parser.parse_args()

    bbox = (args.bbox[1], args.bbox[0], args.bbox[3], args.bbox[2])
    num_prompts = args.num_prompts
    output_file = args.output_file

    generate_prompts(bbox, num_prompts, output_file)
