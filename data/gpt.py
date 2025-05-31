import json
from openai import OpenAI
import os
import copy

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

with open("data/prompt/system_message.txt", "r") as file:
    content = file.read()
system_message = content
with open("data/prompt/example1_context.txt", "r") as file:
    content = file.read()
example1_context = content
with open("data/prompt/example1_response.txt", "r") as file:
    content = file.read()
example1_response = content
with open("data/prompt/example2_context.txt", "r") as file:
    content = file.read()
example2_context = content
with open("data/prompt/example2_response.txt", "r") as file:
    content = file.read()
example2_response = content

prompt_messages = [
    {
        "role": "system",
        "content": system_message,
    },
    {
        "role": "user",
        "content": example1_context,
    },
    {
        "role": "assistant",
        "content": example1_response,
    },
    {
        "role": "user",
        "content": example2_context,
    },
    {
        "role": "assistant",
        "content": example2_response,
    },
]


with open("data/promptGPT.json", "r") as file:
    query_list = json.load(file)


if os.path.exists("data/gpt/gpt4.json"):
    # 如果文件存在，则打开并读取JSON数据
    with open("data/gpt/gpt4.json", "r") as file:
        gpt_list = json.load(file)
else:
    # 如果文件不存在，则创建一个新文件，并写入默认的JSON数据
    with open("data/gpt/gpt4.json", "w") as file:
        gpt_list = []
        json.dump(gpt_list, file)


def save_json(file, name):
    # 指定保存 JSON 的文件路径
    json_file_path = name

    # 将数组转换为 JSON 格式的字符串
    json_data = json.dumps(file, indent=2)  # indent参数是为了可读性，它会添加缩进

    # 将 JSON 数据写入文件
    with open(json_file_path, "w") as json_file:
        json_file.write(json_data)

    print(f"JSON 数据已写入文件: {json_file_path}")


openai_key = "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"
client = OpenAI(api_key=openai_key)

gpt_len = len(gpt_list)
gpt_json = []
for query in query_list[gpt_len:]:
    query_ocr = query["ocr"]
    query_ocr_string = [str(i) for i in query_ocr]
    ocr = "\n".join(query_ocr_string)
    caption = query["caption"]

    context = "\n".join([ocr, caption])
    print(context)

    query_messages = copy.deepcopy(prompt_messages)
    query_messages.append({"role": "user", "content": context})

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106", messages=query_messages
        )
    except Exception as e:
        save_json(gpt_json, "./data/gpt/gpt4-" + str(len(gpt_json)) + ".json")
        break
    else:
        response = completion.choices[0].message
        res_json = {"id": query["id"], "image": query["image"], "response": response}
        gpt_json.append(res_json)
        print(completion.choices[0].message)
        if len(gpt_json) % 100 == 0:
            save_json(gpt_json, "./data/gpt/gpt4-" + str(len(gpt_json)) + ".json")

save_json(gpt_json, "./data/gpt/gpt4.json")
