import os
import json

def add_image_score(json_data, folder_path):
    for item in json_data:
        item_id = item["id"]
        item_folder_path = os.path.join(folder_path, item_id)
        if os.path.isdir(item_folder_path):
            num_files = len(os.listdir(item_folder_path))
            item["image_score"] = num_files - 2

def main():
    input_file_path = "path/to/codes/deita/output/LLaVAR/llava_instruct_150k_llavar_16k_complexity_quality.json"
    output_file_path = "path/to/codes/deita/output/LLaVAR/llava_instruct_150k_llavar_16k_complexity_quality_image.json"
    folder_path = "path/to/codes/segment-anything/output/LLaVAR"

    # Read input JSON file
    with open(input_file_path, "r") as file:
        json_data = json.load(file)

    # Add image score attribute
    add_image_score(json_data, folder_path)

    # Write output JSON file
    with open(output_file_path, "w") as file:
        json.dump(json_data, file, indent=2)

if __name__ == "__main__":
    main()
