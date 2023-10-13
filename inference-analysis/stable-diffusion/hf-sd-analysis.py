import json
import requests
import time
import os

token = os.environ['HUGGING_FACE_TOKEN']

API_URL = "HUGGINGFACE_ENDPOINT_URL"
headers = {
	"Authorization": f"Bearer {token}",
	"Content-Type": "application/json"
}
payload = {
	"inputs": "An astronaut riding a rainbow unicorn, cinematic, dramatic",
}

latency = []
TRIES = 50

for t in range(TRIES):
    start_time = time.time()
    response = requests.post(API_URL, headers=headers, json=payload)
    elapsed_time = time.time() - start_time

    if response.status_code != 200:
        print(response.status_code, response.content)

    print(f'{t}, {elapsed_time:.4f}s')

    latency.append(elapsed_time)

results = {
    'hf' : latency
}

with open('hf-sd-results.json', 'w') as f:
    json.dump(results, f)