import json
import requests
import time
import os

url = "RUNPOD_ENDPOINT_URL"

token = os.environ['RUNPOD_TOKEN']

headers = {
  'Content-Type': 'application/json',
  'Authorization': f'Bearer {token}'
}

prompt = "An astronaut riding a rainbow unicorn, cinematic, dramatic"

# payload = json.dumps({
#             "prompt": prompt
#         })

payload = json.dumps({"input" :{"prompt" : prompt}})


TRIES = 50
latency = []

for i in range(TRIES):
    start_time = time.time()
    response = requests.request("POST", url=url, headers=headers, data=payload)
    elapsed_time = time.time() - start_time


    latency.append(elapsed_time)

    print(i, elapsed_time)

with open('runpod-sd-results.json', 'w') as f:
    json.dump({"runpod" : latency}, f)
