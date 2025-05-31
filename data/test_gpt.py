from openai import OpenAI
import os

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "https://localhost:7890"

openai_key = "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"
client = OpenAI(api_key=openai_key)


completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {
            "role": "system",
            "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
        },
        {
            "role": "user",
            "content": "Compose a poem that explains the concept of recursion in programming.",
        },
    ],
)
res = completion
print(completion.choices[0].message)
print(completion.choices[0].message)
