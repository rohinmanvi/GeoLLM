# GeoLLM

This is the official repository for ["_GeoLLM: Extracting Geospatial Knowledge from Large Language Models_"](https://arxiv.org/abs/2310.06213) (ICLR 2024) and ["_Large Language Models are Geographically Biased_"](https://arxiv.org/abs/2402.02680).

Authors: 
[Rohin Manvi](https://www.linkedin.com/in/rohin-manvi-2a9226187/) <sup>1</sup>,
[Samar Khanna](https://samar-khanna.github.io), 
[Gengchen Mai](https://gengchenmai.github.io/),
[Marshall Burke](https://web.stanford.edu/~mburke/), 
[David B. Lobell](https://earth.stanford.edu/people/david-lobell#gs.5vndff), 
[Stefano Ermon](https://cs.stanford.edu/~ermon/).

<sub><sup>1</sup> Corresponding author, rohinm@cs.stanford.edu.</sub>

---

### Pregenerated GeoLLM prompts

There are 100,000 prompts generated for locations around the world in the `prompts/100000_prompts.jsonl` file. You can select a subset of these to make predictions. For example, there are 2,000 prompts selected for visualizations of the world in the `prompts/world_prompts.jsonl` file. Additionally, prompts can be generated at a much higher resolution. For example, there are 2,000 prompts for the Bay Area in the `prompts/bay_area_prompts.jsonl` file.

You can use the `select_visualization_prompts.py` script to select a subset of prompts from an input file for visualization. This script uses importance sampling and farthest point sampling to select a maximum of `MAX_NUM_PROMPTS` prompts that contain at least one of the region names. The sampling is used to ensure that the selected prompts represent relevant locations and are geographically spread out.

```shell
python3 select_visualization_prompts.py <INPUT_PROMPTS_FILE> <OUTPUT_PROMPTS_FILE> <MAX_NUM_PROMPTS> <REGION_1_NAME> <REGION_2_NAME> ...
```

Where:
- `<INPUT_PROMPTS_FILE>` is the path to the input prompts jsonl file (e.g. "prompts/100000_prompts.jsonl").
- `<OUTPUT_PROMPTS_FILE>` is the path to where the prompts should be written (e.g. "prompts/selected_prompts.jsonl").
- `<MAX_NUM_PROMPTS>` is the maximum number of prompts to select.
- `<REGION_1_NAME>`, `<REGION_2_NAME>`, ... are the names of the regions you want to include in the selected prompts.

Examples:

```shell
python3 select_visualization_prompts.py prompts/100000_prompts.jsonl prompts/world_prompts.jsonl 2000 ""
```

```shell
python3 select_visualization_prompts.py prompts/bay_area_prompts.jsonl prompts/bay_area_except_north_bay_prompts.jsonl 1000 "San Francisco, CAL Fire Northern Region" "San Mateo County" "Santa Clara County" "Alameda County"
```

### Generating more GeoLLM prompts

If you want to generate your own prompts and have a list of coordinates, you can use the `generate_geollm_prompts_with_csv.py` script.

```shell
python3 generate_geollm_prompts_with_csv.py <CSV_FILE_WITH_COORDINATES>
```

Where `<CSV_FILE_WITH_COORDINATES>` is a csv file containing the coordinates. It should have a header with `Latitude` and `Longitude` columns. The script will generate prompts for each pair of coordinates and write them to a file with the same name in the `prompts` folder.

If you want to generate prompts for a specific region in a bounding box, you can use the `generate_geollm_prompts_at_location.py` script. It uses the same sampling method as the `select_visualization_prompts.py` script to select prompts for a specific region.

```shell
python3 generate_geollm_prompts_at_location.py <OUTPUT_PROMPTS_FILE> <MAX_NUM_PROMPTS> <MIN_LATITUDE> <MIN_LONGITUDE> <MAX_LATITUDE> <MAX_LONGITUDE>
```

Where:
- `<OUTPUT_PROMPTS_FILE>` is the path to where the prompts should be written (e.g. "prompts/new_prompts.jsonl").
- `<MAX_NUM_PROMPTS>` is the maximum number of prompts to select.
- `<MIN_LATITUDE>`, `<MIN_LONGITUDE>`, `<MAX_LATITUDE>`, `<MAX_LONGITUDE>` are the coordinates of the bounding box.

An example:

```shell
python3 generate_geollm_prompts_at_location.py prompts/bay_area_prompts.jsonl 2000 37.13930393009039 -122.54505349168528 38.03830072195632 -121.78355363422295
```

---

### Zero-shot predictions (no fine-tuning required)

You can use the `make_predictions_and_visualize.py` script to make zero-shot predictions with any LLM of your choice from the OpenAI, Google, or Together APIs. Please note that while zero-shot predictions can be quite accurate, they can contain biases, especially for subjective topics (as shown in ["_Large Language Models are Geographically Biased_"](https://arxiv.org/abs/2402.02680)). Outputs will be in csv and html formats in the results folder for predictions and visualization, respectively.

```shell
python3 make_predictions_and_visualize.py <API> <API_KEY> <MODEL_NAME> <PROMPTS_FILE> <TASK_NAME>
```

Where: 
- `<API>` is one of `openai`, `google`, or `together`.
- `<API_KEY>` is the API key for the chosen API.
- `<MODEL_NAME>` is the name of the LLM model you want to use (e.g. "gpt-3.5-turbo-0613").
- `<PROMPTS_FILE>` is the path to the file with geollm prompts (e.g. "prompts/world_prompts.jsonl").
- `<TASK_NAME>` is the name of the task you want to make predictions for (e.g. "Infant Mortality Rate").

For example, to make zero-shot predictions for "Infant Mortality Rate" around the world using OpenAI's GPT-3.5-turbo, you can use the following command:

```shell
python3 make_predictions_and_visualize.py openai sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX gpt-3.5-turbo-0613 prompts/world_prompts.jsonl "Infant Mortality Rate"
```

The predictions would be in `results/gpt_3_5_turbo_0613_Infant_Mortality_Rate_world_prompts.csv` and the visualization would be in `results/gpt_3_5_turbo_0613_Infant_Mortality_Rate_world_prompts.html`. There can also be versions with the expected value (w/ logprobs) predictions if using OpenAI's API.

### Fine-tuning for higher quality data extraction

If you need to extract high-quality geospatial data and have access to a sample of ground truth data, you can use the `generate_fine_tuning_data.py` script to generate a fine-tuning dataset for OpenAI's finetuning API (https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset). This dataset can then be used to create a finetuned version of GPT-3.5. You can also use it to finetune other LLMs, but you will need to modify the dataset and finetune the model yourself.

```shell
python3 create_gpt_finetuning_data.py <TASK_NAME> <CSV_FILE_WITH_GROUNDTRUTH> <PROMPTS_FILE>
```

Where:
- `<TASK_NAME>` is the name of the task you have ground truth for (e.g. "Population Density").
- `<CSV_FILE_WITH_GROUNDTRUTH>` is the path to the csv file with ground truth data for the task.
- `<PROMPTS_FILE>` is the path to the file with geollm prompts for each ground truth data point.

Once you have a fine-tuned OpenAI model, simply use the finetuned model with the `make_predictions_and_visualize.py` script.

---

### Evaluating performance

If you have access to a GeoTIFF file with ground truth data, you can use the `evaluate_predictions.py` script to evaluate the performance of the predictions made by the LLM. This script will extract the corresponding ground truth data from the GeoTIFF file and calculate the spearman correlation between the predictions and the ground truth data.

```shell
python3 calculate_spearman_correlation.py <PREDICTIONS_CSV_FILE> <GEOTIFF_FILE>
```

Where:
- `<PREDICTIONS_CSV_FILE>` is the path to the csv file with the predictions (found in `results` folder).
- `<GEOTIFF_FILE>` is the path to the GeoTIFF file with the ground truth data.

An example:

```shell
python3 calculate_spearman_correlation.py results/gpt_3_5_turbo_0613_Infant_Mortality_Rate_world_prompts_expected_value.csv data/povmap_global_subnational_infant_mortality_rates_v2_01.tif
```

### Evaluating biases

To evaluate the biases in the predictions made by the LLM, you can use the `calculate_bias_score.py` script. This script will calculate the bias score as defined in our paper. This should only be used with predictions on sensitive subjective topics, as the bias score is only meaningful for such topics.

```shell
python3 calculate_bias_score.py <PREDICTIONS_CSV_FILE> <GEOTIFF_FILE> <NUM_PROMPTS>
```

Where:
- `<PREDICTIONS_CSV_FILE>` is the path to the csv file with the predictions (found in `results` folder).
- `<GEOTIFF_FILE>` is the path to the GeoTIFF file with the anchoring bias distribution data.
- `<NUM_PROMPTS>` is the number of prompts used to make the predictions.

An example:

```shell
python3 calculate_bias_score.py results/gpt_3_5_turbo_0613_Average_Attractiveness_of_Residents_world_prompts_expected_value.csv data/povmap_global_subnational_infant_mortality_rates_v2_01.tif 2000
```

Note that the bias score can be negative if the predictions are negatively correlated with the anchoring bias distribution. This indicates that the predictions are biased in the opposite direction of with respect to the anchoring bias distribution. In this case, it would indicate that the predictions are biased towards infant _survival_ rates.

## Citation
If you found GeoLLM helpful, please cite our papers (second paper enabled zero-shot predictions and evaluated biases):
```
@inproceedings{manvi2024geollm,
      title={Geo{LLM}: Extracting Geospatial Knowledge from Large Language Models},
      author={Rohin Manvi and Samar Khanna and Gengchen Mai and Marshall Burke and David B. Lobell and Stefano Ermon},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://openreview.net/forum?id=TqL2xBwXP3}
}

@misc{manvi2024large,
      title={Large Language Models are Geographically Biased}, 
      author={Rohin Manvi and Samar Khanna and Marshall Burke and David Lobell and Stefano Ermon},
      year={2024},
      eprint={2402.02680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
