import json

def merge_json_files(file1_path, file2_path, output_path):
    # 读取第一个JSON文件
    with open(file1_path, 'r') as file1:
        data1 = json.load(file1)

    # 读取第二个JSON文件
    with open(file2_path, 'r') as file2:
        data2 = json.load(file2)

    # 创建一个字典用于存储合并后的数据
    merged_data = {}

    # 合并第一个文件的数据
    for item in data1:
        id_value = item.get('id')
        merged_data[id_value] = item

    # 合并第二个文件的数据，保留相同的属性，合并不同的属性
    for item in data2:
        id_value = item.get('id')
        if id_value in merged_data:
            # 合并属性
            merged_data[id_value].update(item)
        else:
            # 如果ID在第一个文件中不存在，则直接添加到合并后的数据中
            merged_data[id_value] = item

    # 将合并后的数据写入输出文件
    with open(output_path, 'w') as output_file:
        json.dump(list(merged_data.values()), output_file, indent=2)

# 使用示例
merge_json_files('data/ocr_llavar_finetune.json', 'data/caption_llavar_finetune_complex.json', 'data/promptGPT.json')
