import rasterio
import jsonlines

PREFIX = """You will be given data about a specific location randomly sampled from all human-populated locations on Earth.
You give your rating keeping in mind that it is relative to all other human-populated locations on Earth (from all continents, countries, etc.).
You provide ONLY your answer in the exact format "My answer is X.X." where 'X.X' represents your rating for the given topic.


"""

ADJACENT_PIXELS = 12

def load_geollm_prompts(file_path, task):
    with jsonlines.open(file_path, 'r') as reader:
        data = list(reader)
    geollm_prompts = [PREFIX + item['text'].strip().replace("<TASK>", task) for item in data]
    return geollm_prompts

def get_coordinates(text):
    text = text.split("Coordinates: ")[1]
    coordinates = text[text.find('(')+1:text.find(')')].split(", ")
    lat, lon = list(map(float, coordinates))
    return lat, lon

def normalized_fractional_ranking(numbers):
    sorted_numbers = sorted(enumerate(numbers), key=lambda x: x[1])

    ranks = {}
    for rank, (original_index, number) in enumerate(sorted_numbers):
        if number in ranks:
            ranks[number][0] += rank + 1
            ranks[number][1] += 1
        else:
            ranks[number] = [rank + 1, 1]

    average_ranks = {number: total_rank / count for number, (total_rank, count) in ranks.items()}

    return [(average_ranks[number] - 1) / len(numbers) for number in numbers]

def extract_data(lat, lon, file_path):
    with rasterio.open(file_path) as src:
        transform = ~src.transform
        x, y = transform * (lon, lat)
        px, py = round(x), round(y)

        window = ((max(py-ADJACENT_PIXELS, 0), min(py+ADJACENT_PIXELS+1, src.height)), (max(px-ADJACENT_PIXELS, 0), min(px+ADJACENT_PIXELS+1, src.width)))
        data = src.read(1, window=window)
        non_negative_data = data[data >= 0]
        total_population = non_negative_data.sum()

        return total_population if 0 <= px < src.width and 0 <= py < src.height else None