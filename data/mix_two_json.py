import json

# 读取第一个JSON文件
with open("/home/zxy/data/MY/gptqa_15k_14993.json", "r") as file:
    json_data1 = json.load(file)

# 读取第二个JSON文件
with open("/home/zxy/data/LLaVAR/finetune/llava_instruct_150k_llavar_16k.json", "r") as file:
    json_data2 = json.load(file)

# 合并两个JSON数据
merged_data = json_data1 + json_data2

# 将合并后的数据写入新的JSON文件
with open("/home/zxy/data/MY/llava_instruct_150k_llavar_16k_gptqa_15k.json", "w") as file:
    json.dump(merged_data, file, indent=2, ensure_ascii=False)
