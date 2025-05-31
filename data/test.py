import json
with open("data/promptGPT.json", "r", encoding="UTF-8") as file:
    query_list = json.load(file)
print(len(query_list))