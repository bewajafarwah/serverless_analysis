import json
import pandas as pd
import requests
import time

url = "MODAL_ENDPOINT_URL"

headers = {
  'Content-Type': 'application/json',
}

prompt = "An astronaut riding a rainbow unicorn, cinematic, dramatic"

payload = json.dumps({
            "prompt": prompt
        })


TRIES = 50
latency = []

for i in range(TRIES):
    start_time = time.time()
    response = requests.request("POST", url=url, headers=headers, data=payload)
    elapsed_time = time.time() - start_time

    latency.append(elapsed_time)

    print(i, elapsed_time)

with open('modal-sd-results.json', 'w') as f:
    json.dump({"modal" : latency}, f)
